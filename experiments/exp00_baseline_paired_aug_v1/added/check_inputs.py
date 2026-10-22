"""CPU-only input audit. Never import/construct a network or load weights."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def capture(dataset, seed, workers, epochs, perturb):
    import random
    import albumentations as A
    import numpy as np
    import torch
    from albumentations.pytorch import ToTensorV2
    from paired_inputs import PROTOCOL, PairedDataset, make_loader, set_loader_epoch, InputDigest

    torch.set_num_threads(1)
    # Load the ACTUAL, unchanged data factory function bodies from the training
    # file, without its top-level timm/network/metric imports or main routine.
    names = {'get_train_transform', 'get_valid_transform', 'load_ids'}
    tree = ast.parse((HERE / 'kvasir_train.py').read_text())
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                and n.name in names], type_ignores=[])
    if {n.name for n in functions.body} != names:
        raise RuntimeError('Training data factory functions not found')
    namespace = dict(A=A, ToTensorV2=ToTensorV2, os=os)
    exec(compile(functions, str(HERE / 'kvasir_train.py'), 'exec'), namespace)
    config = json.loads((HERE / 'config.json').read_text())['train_options']
    random.seed(perturb)
    np.random.seed(perturb)
    torch.manual_seed(perturb)
    loaders = {}
    sizes = {}
    for phase in ('train', 'valid'):
        split = 'train' if phase == 'train' else 'val'
        ids = namespace['load_ids'](str(HERE / f'configs/splits/kvasir/{split}.txt'))
        sizes[phase] = len(ids)
        factory = namespace['get_train_transform' if phase == 'train' else 'get_valid_transform']
        data = PairedDataset(dataset, ids, factory(config['img_size']), seed, phase)
        loaders[phase] = make_loader(data, config['batch'] if phase == 'train'
                                    else max(1, config['batch'] // 2), workers)
    rows, probes = [], []
    for epoch in range(epochs):
        for phase, loader in loaders.items():
            # Model/global RNG consumption must not change shuffle or Compose.
            torch.rand(11 + perturb)
            random.random()
            np.random.rand(7 + perturb)
            set_loader_epoch(loader, epoch)
            digest = InputDigest()
            for images, masks, ids in loader:
                digest.update(images, masks, ids)
            rows.append(digest.record(seed, epoch, phase))
            if phase == 'train':
                # Fixed sample IDs distinguish changed AUGMENTATIONS from a
                # mere change in shuffled order. No image cache is written.
                for index in range(min(8, len(loader.dataset))):
                    image, mask, sample_id = loader.dataset[index]
                    probe = InputDigest()
                    probe.update(image.unsqueeze(0), mask.unsqueeze(0), [sample_id])
                    probes.append(dict(sample_id=sample_id, **probe.record(seed, epoch, phase)))
    return dict(protocol=PROTOCOL, seed=seed, workers=workers, dataset_sizes=sizes, records=rows,
                augmentation_probes=probes,
                versions=dict(python=sys.version, torch=torch.__version__, albumentations=A.__version__,
                              numpy=np.__version__))


def validate(captures):
    first, repeated, single_worker, different_seed = captures
    if not first['records'] == repeated['records'] == single_worker['records']:
        raise RuntimeError('FAIL: independent processes / worker counts have different inputs')
    if not first['augmentation_probes'] == repeated['augmentation_probes'] == single_worker['augmentation_probes']:
        raise RuntimeError('FAIL: fixed-sample augmentations differ across processes')
    train = [r for r in first['records'] if r['phase'] == 'train']
    if len({r['order_sha256'] for r in train}) < 2:
        raise RuntimeError('FAIL: training sample order did not change across epochs')
    if len({r['images_sha256'] for r in train}) < 2:
        raise RuntimeError('FAIL: training sequence did not change across epochs')
    changed = [r for r in different_seed['records'] if r['phase'] == 'train']
    if all(a['images_sha256'] == b['images_sha256'] for a, b in zip(train, changed)):
        raise RuntimeError('FAIL: changed seed produced the same training sequence')
    probes = first['augmentation_probes']
    anchor = {p['sample_id']: p['images_sha256'] for p in probes if p['epoch'] == 0}
    if all(p['images_sha256'] == anchor[p['sample_id']] for p in probes if p['epoch'] > 0):
        raise RuntimeError('FAIL: fixed-sample probes did not change augmentation across epochs')
    if all(a['images_sha256'] == b['images_sha256'] for a, b in
           zip(probes, different_seed['augmentation_probes'])):
        raise RuntimeError('FAIL: fixed-sample augmentations did not change with seed')
    for result in captures:
        for row in result['records']:
            if not row['batches'] or not row['samples']:
                raise RuntimeError('FAIL: empty input audit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='/data/dataset/zsy/mslau-net/Kvasir-SEG')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--perturb', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.epochs < 2 or args.workers < 0:
        parser.error('Use at least two epochs and nonnegative workers')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.capture:
        result = capture(args.dataset, args.seed, args.workers, args.epochs, args.perturb)
    else:
        captures = []
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', NO_ALBUMENTATIONS_UPDATE='1')
        with tempfile.TemporaryDirectory(prefix='paired_input_capture_') as temporary:
            for index, (workers, seed, perturb) in enumerate([
                    (args.workers, args.seed, 0), (args.workers, args.seed, 13),
                    (0, args.seed, 29), (0, args.seed + 1, 41)]):
                report = Path(temporary) / f'{index}.json'
                print(f'CPU input capture {index + 1}/4: workers={workers}, seed={seed}', flush=True)
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--capture',
                                '--dataset', args.dataset, '--seed', str(seed), '--workers', str(workers),
                                '--epochs', str(args.epochs), '--perturb', str(perturb),
                                '--output', str(report)], env=environment, cwd=HERE, check=True)
                captures.append(json.loads(report.read_text()))
        validate(captures)
        result = dict(status='passed', input_protocol='paired_aug_v1', dataset=str(Path(args.dataset).resolve()),
                      captures=captures, source=json.loads((HERE / 'build_manifest.json').read_text()),
                      notes=['Input equality only; no network/GPU computation or training occurred.',
                             'Cross-model input equality still needs checking when NAF adopts this protocol.'])
        print('PASS: independent launches and 0/requested workers match; epoch/seed changes verified.', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Report:', args.output, flush=True)


if __name__ == '__main__':
    main()
