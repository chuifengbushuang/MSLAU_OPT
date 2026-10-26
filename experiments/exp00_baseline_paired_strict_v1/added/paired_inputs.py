"""Sample/epoch-keyed input randomness; independent of model and worker RNGs."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, SequentialSampler

from loader import binary_class

PROTOCOL = 'paired_aug_v1'


def stable_seed(seed, stream, *parts):
    payload = json.dumps([PROTOCOL, int(seed), stream, *parts],
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & ((1 << 63) - 1)


class PairedDataset(binary_class):
    """Keep the original reader/mask conversion; only seed its Compose call."""
    def __init__(self, path, data, transform, seed, phase):
        super().__init__(path, data, transform)
        self.seed, self.phase, self.epoch = int(seed), phase, 0
        if not hasattr(transform, 'set_random_seed'):
            raise TypeError('This protocol requires Compose.set_random_seed (Albumentations 1.4.24)')

    def __getitem__(self, index):
        sample_id = self.folders[index]
        self.transforms.set_random_seed(stable_seed(
            self.seed, 'augmentation', self.phase, self.epoch, sample_id))
        return super().__getitem__(index)


class EpochRandomSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        generator = torch.Generator().manual_seed(stable_seed(
            self.dataset.seed, 'sample_order', self.dataset.epoch))
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)


def seed_worker(worker_id):
    # These seeds cover other code using global RNGs. Compose is seeded per
    # sample above, NOT from worker_id or a worker-local augmentation stream.
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size, num_workers, pin_memory=False):
    train = dataset.phase == 'train'
    loader = DataLoader(dataset, batch_size=batch_size,
                        sampler=EpochRandomSampler(dataset) if train else SequentialSampler(dataset),
                        drop_last=train, num_workers=num_workers, pin_memory=pin_memory,
                        worker_init_fn=seed_worker, generator=torch.Generator(),
                        persistent_workers=False)
    set_loader_epoch(loader, 0)
    return loader


def set_loader_epoch(loader, epoch):
    if loader.persistent_workers:
        raise ValueError('Set epoch before iteration; paired_aug_v1 requires persistent_workers=False')
    dataset = loader.dataset
    dataset.epoch = int(epoch)
    loader.generator.manual_seed(stable_seed(dataset.seed, 'worker_base', dataset.phase, int(epoch)))


def update_tensor_digest(digest, tensor):
    value = tensor.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
    digest.update(len(header).to_bytes(8, 'big'))
    digest.update(header)
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))


class InputDigest:
    """Streaming CPU hashes include batch boundaries, IDs, shapes and dtypes."""
    def __init__(self):
        self.order, self.images, self.masks = (hashlib.sha256() for _ in range(3))
        self.batches = self.samples = 0

    def update(self, images, masks, ids):
        encoded = json.dumps(list(ids), ensure_ascii=False, separators=(',', ':')).encode()
        self.order.update(len(encoded).to_bytes(8, 'big'))
        self.order.update(encoded)
        update_tensor_digest(self.images, images)
        update_tensor_digest(self.masks, masks)
        self.batches += 1
        self.samples += len(ids)

    def record(self, seed, epoch, phase):
        return dict(protocol=PROTOCOL, seed=int(seed), epoch=int(epoch), phase=phase,
                    batches=self.batches, samples=self.samples, order_sha256=self.order.hexdigest(),
                    images_sha256=self.images.hexdigest(), masks_sha256=self.masks.hexdigest())


def append_record(path, record):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')


def state_digest(state):
    """Hash tensor contents, not torch.save archives (which can differ in bytes)."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        encoded = name.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)
        update_tensor_digest(digest, value)
    return digest.hexdigest()


def record_initial_state(model, seed, log_dir):
    record = dict(protocol=PROTOCOL, seed=int(seed), state_sha256=state_digest(model.state_dict()),
                  cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  cudnn_benchmark=torch.backends.cudnn.benchmark)
    with (Path(log_dir) / 'initial_state.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    return record
