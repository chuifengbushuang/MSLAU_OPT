"""Deterministic, decoupled per-sample augmentation and epoch-wise sampling.

Corrected ("v2") variant of the paired-input protocol:
  - stable SHA256-based seeds from (seed, phase, epoch, sample_id) -- no
    Python hash;
  - each augmentation sub-transform receives its OWN derived seed, so the
    geometric / photometric operators are NOT coupled. (The v1 helper reset
    the whole Compose with the SAME seed per sample, which made the p=0.25
    flips/affine fire together ~25% of the time; that defect is not reused.)
  - per-epoch RandomSampler backed by an independent torch.Generator.

Fully deterministic given (seed, phase, epoch, sample order).
"""

import hashlib
import json
import random

import numpy as np
import torch
from torch.utils.data import Sampler

from loader import binary_class

_NAMESPACE = "paired_v2_corrected"


def stable_seed(*parts):
    payload = json.dumps(
        [_NAMESPACE] + list(parts),
        separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 63 - 1)


def _seed_globals(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed % (2 ** 63))


class PairedDataset(binary_class):
    """binary_class with deterministic, decoupled per-sample augmentation."""

    def __init__(self, path, data, transform=None, seed=1234,
                 phase="train", epoch=0):
        super().__init__(path, data, transform)
        self.seed = int(seed)
        self.phase = phase
        self.epoch = int(epoch)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _seed_pipeline(self, sample_id):
        base = stable_seed(self.seed, self.phase, self.epoch, sample_id)
        transforms = getattr(self.transforms, "transforms", None)
        if transforms:
            for index, transform in enumerate(transforms):
                if hasattr(transform, "set_random_seed"):
                    transform.set_random_seed(
                        stable_seed(base, "op", index) % (2 ** 32))
        _seed_globals(base)

    def __getitem__(self, index):
        self._seed_pipeline(self.folders[index])
        return super().__getitem__(index)


class EpochRandomSampler(Sampler):
    """Per-epoch shuffle with an independent generator (deterministic order)."""

    def __init__(self, data_source, seed=1234, phase="train"):
        self.data_source = data_source
        self.seed = int(seed)
        self.phase = phase
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(
            stable_seed(self.seed, self.phase, self.epoch, "order"))
        order = torch.randperm(len(self.data_source), generator=generator)
        return iter(order.tolist())

    def __len__(self):
        return len(self.data_source)


def make_worker_init_fn(seed, phase, epoch_provider):
    def worker_init_fn(worker_id):
        epoch = epoch_provider() if callable(epoch_provider) else 0
        _seed_globals(stable_seed(seed, "worker", phase, epoch, worker_id))
    return worker_init_fn
