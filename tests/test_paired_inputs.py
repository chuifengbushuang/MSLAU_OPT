"""Real augmentation/multiprocess checks; CPU fixtures, never formal training."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/exp00_baseline'
EXPERIMENT = 'exp00_baseline_paired_aug_v1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                               and n.name in names], type_ignores=[])
    exec(compile(selected, str(path), 'exec'), namespace)
    return namespace


try:
    import albumentations as A
    import numpy as np
    import torch
    from albumentations.pytorch import ToTensorV2
    from PIL import Image
    from torch import nn
    from torch.autograd import Variable
except ImportError:
    A = None


class PairedDefinitionTests(unittest.TestCase):
    def test_config_and_all_frozen_files_preserved(self):
        original = json.loads((BASE / 'config.json').read_text())
        actual = json.loads((ROOT / 'experiments' / EXPERIMENT / 'config.json').read_text())
        actual['id'] = original['id']
        self.assertEqual(actual.pop('input_protocol'), 'paired_aug_v1')
        self.assertEqual(actual, original)
        manifest = json.loads((BASE / 'source_manifest.json').read_text())
        for name, sha in manifest['files'].items():
            self.assertEqual(hashlib.sha256((BASE / name).read_bytes()).hexdigest(), sha)


@unittest.skipIf(A is None, 'real albumentations / torch / skimage required')
class PairedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix='paired_unit_')
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / 'source'
        builder = load('paired_builder', ROOT / 'tools/build_experiment.py')
        cls.manifest = builder.build(EXPERIMENT, cls.source)
        sys.path.insert(0, str(cls.source))
        cls.data = load('paired_inputs', cls.source / 'paired_inputs.py')
        cls.factories = definitions(cls.source / 'kvasir_train.py',
                                   {'get_train_transform', 'get_valid_transform'},
                                   dict(A=A, ToTensorV2=ToTensorV2))
        cls.dataset = cls.root / 'data'
        (cls.dataset / 'images').mkdir(parents=True)
        (cls.dataset / 'masks').mkdir()
        cls.ids = [f'{index}.png' for index in range(13)]
        for index, name in enumerate(cls.ids):
            rng = np.random.default_rng(index)
            image = rng.integers(0, 256, (37, 43, 3), dtype=np.uint8)
            mask = np.zeros((37, 43), dtype=np.uint8)
            mask[4 + index % 7:28, 7:29 + index % 8] = 255
            Image.fromarray(image).save(cls.dataset / 'images' / name)
            Image.fromarray(mask).save(cls.dataset / 'masks' / name)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.source))
        # The frozen loader name must not leak into tests of other experiments.
        sys.modules.pop('loader', None)
        cls.temporary.cleanup()

    def loader(self, seed=1234, workers=0, phase='train', size=32):
        factory = self.factories['get_train_transform' if phase == 'train' else 'get_valid_transform']
        dataset = self.data.PairedDataset(str(self.dataset), self.ids, factory(size), seed, phase)
        return self.data.make_loader(dataset, 4, workers)

    def collect(self, loader, epoch):
        self.data.set_loader_epoch(loader, epoch)
        digest = self.data.InputDigest()
        ids_all = []
        for images, masks, ids in loader:
            ids_all.extend(ids)
            digest.update(images, masks, ids)
        return digest.record(loader.dataset.seed, epoch, loader.dataset.phase), ids_all

    def test_model_global_rng_does_not_change_order_or_augmentation(self):
        torch.manual_seed(1234)
        before = torch.get_rng_state()
        first = self.loader()
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
        a, _ = self.collect(first, 0)
        torch.rand(431)
        np.random.rand(23)
        random.random()
        b, _ = self.collect(self.loader(), 0)
        self.assertEqual(a, b)
        self.assertEqual(a['samples'], 12)  # same batch/drop_last policy

    def test_workers_zero_and_eight_match_across_epochs(self):
        zero, eight = self.loader(), self.loader(workers=8)
        for epoch in (0, 1):
            self.assertEqual(self.collect(zero, epoch), self.collect(eight, epoch))

    def test_augmentations_are_keyed_to_sample_not_access_order_or_epoch_copy(self):
        loader = self.loader()
        data = loader.dataset
        a = data[0]
        data[7]
        b = data[0]
        torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
        torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
        original = [data[i][0] for i in range(8)]
        self.data.set_loader_epoch(loader, 1)
        changed = [data[i][0] for i in range(8)]
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(original, changed)))
        other = self.loader(seed=42).dataset
        self.assertTrue(any(not torch.equal(x, other[i][0]) for i, x in enumerate(original)))

    def test_geometry_keeps_image_and_mask_aligned(self):
        mask = np.zeros((27, 35), np.uint8)
        mask[3:11, 7:21] = 1
        image = np.repeat(mask[:, :, None], 3, axis=2)
        transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                               A.RandomRotate90(p=1)])
        for seed in (1, 42, 1234):
            transform.set_random_seed(seed)
            out = transform(image=image, mask=mask)
            np.testing.assert_array_equal(out['image'][:, :, 0], out['mask'])

    def test_validation_is_constant_and_model_init_rng_is_preserved(self):
        torch.manual_seed(1234)
        before = torch.get_rng_state()
        loader = self.loader(phase='valid')
        first, _ = self.collect(loader, 0)
        second, _ = self.collect(loader, 1)
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
        for field in ('order_sha256', 'images_sha256', 'masks_sha256'):
            self.assertEqual(first[field], second[field])
        original = nn.Conv2d(3, 4, 3)
        torch.set_rng_state(before)
        controlled = nn.Conv2d(3, 4, 3)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, controlled.state_dict()[name], atol=0, rtol=0)

    def test_source_algorithms_and_augmentation_functions_unchanged(self):
        manifest = json.loads((BASE / 'source_manifest.json').read_text())
        for name in manifest['files']:
            if name != 'kvasir_train.py':
                self.assertEqual((BASE / name).read_bytes(), (self.source / name).read_bytes())
        original = ast.parse((BASE / 'kvasir_train.py').read_text())
        controlled = ast.parse((self.source / 'kvasir_train.py').read_text())
        for name in ('get_train_transform', 'get_valid_transform', 'load_ids'):
            a = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in controlled.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(b))

    def test_launcher_preserves_commands_records_and_repeat_labels(self):
        launcher = load('paired_launcher', self.source / 'run.py')
        original = load('paired_original_launcher', BASE / 'run.py')
        train, evaluation = launcher.commands(1234, 'preview')
        old_train, old_evaluation = original.commands(1234, 'preview')
        # All options other than expanded source/split paths are identical.
        self.assertEqual(train[15:], old_train[15:])
        self.assertEqual(evaluation[11:], old_evaluation[11:])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            launched = []

            class Child:
                def __init__(self, command, **kwargs):
                    launched.append(command)
                    if '--output_dir' in command:
                        directory = Path(command[command.index('--output_dir') + 1])
                        for name in ('kvasir_best_model.pth', 'best_iou_named.pth', 'best_loss_named.pth'):
                            (directory / name).write_bytes(b'fixture')
                        logs = Path(command[command.index('--log_dir') + 1])
                        logs.mkdir()
                        (logs / 'initial_state.json').write_text(json.dumps(dict(state_sha256='fixture')))
                        (logs / 'input_fingerprints.jsonl').write_text(''.join(json.dumps(dict(epoch=e, phase=p)) + '\n'
                            for e in range(20) for p in ('train', 'valid')))
                        (logs / 'epoch_metrics.jsonl').write_text('{}\n' * 40)
                        self.stdout = ['Best val IoU: 0.87 at epoch 19\n']
                    else:
                        self.stdout = ['mean IoU: 0.87\n']
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def wait(self):
                    return 0

            with patch('sys.argv', ['run.py', '--seed', '1234', '--run']), \
                 patch.dict(os.environ, {'P51_REPEAT': '2'}), \
                 patch.object(launcher, 'OUTPUT', output), \
                 patch.object(launcher, 'DATASET', str(self.dataset)), \
                 patch.object(launcher, 'HIERA', __file__), \
                 patch.object(launcher.subprocess, 'check_output', return_value='fixture\n'), \
                 patch.object(launcher.subprocess, 'Popen', side_effect=Child), \
                 contextlib.redirect_stdout(io.StringIO()):
                launcher.main()
            manifest = json.loads(next(output.glob('experiments/*/manifest.json')).read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['repeat_label'], '2')
            self.assertEqual(manifest['seed'], 1234)
            self.assertTrue(manifest['run_id'].endswith('_r2'))
            self.assertEqual(len(manifest['checkpoint_files']), 3)
            self.assertEqual(len(launched), 2)
            self.assertEqual(manifest['input_protocol'], 'paired_aug_v1')

    def test_real_training_loop_records_inputs_metrics_and_three_checkpoints(self):
        names = {'train_model', 'snapshot_state_dict', 'atomic_save_state_dict', 'log_fusion_scale_stats'}
        namespace = definitions(self.source / 'kvasir_train.py', names,
            dict(torch=torch, nn=nn, Variable=Variable, copy=copy, time=time, os=os,
                 logging=logging, **{k: getattr(self.data, k) for k in
                     ('set_loader_epoch', 'InputDigest', 'append_record')}))

        class TinyModel(nn.Module):
            decoder_mode = 'legacy'
            def __init__(self):
                super().__init__()
                self.conv_mla = nn.Identity()
                self.conv = nn.Conv2d(3, 1, 1)
            def forward(self, images, **kwargs):
                return self.conv(images)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'logs').mkdir()
            model = TinyModel()
            initial = self.data.record_initial_state(model, 1234, directory / 'logs')
            self.assertEqual(initial['state_sha256'], self.data.state_digest(model.state_dict()))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
            loaders = {phase: self.loader(phase=phase) for phase in ('train', 'valid')}
            with patch('torch.cuda.is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
                namespace['train_model'](model, nn.BCEWithLogitsLoss(), optimizer, scheduler,
                    loaders, lambda a, b: (a.sigmoid() - b).abs().mean(), 2, tmp,
                    input_audit_path=str(directory / 'logs/input_fingerprints.jsonl'))
            rows = [json.loads(line) for line in (directory / 'logs/input_fingerprints.jsonl').read_text().splitlines()]
            self.assertEqual([(r['epoch'], r['phase']) for r in rows],
                             [(0, 'train'), (0, 'valid'), (1, 'train'), (1, 'valid')])
            self.assertEqual(len((directory / 'logs/epoch_metrics.jsonl').read_text().splitlines()), 4)
            self.assertEqual(len(list(directory.glob('*.pth'))), 3)
            expected, _ = self.collect(self.loader(), 0)
            self.assertEqual(rows[0], expected)

    def test_cpu_checker_uses_independent_processes_without_network_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            import shutil
            shutil.copytree(self.source, source, ignore=shutil.ignore_patterns('__pycache__'))
            config = json.loads((source / 'config.json').read_text())
            config['train_options'].update(img_size=32, batch=4)
            (source / 'config.json').write_text(json.dumps(config))
            for phase in ('train', 'val'):
                (source / f'configs/splits/kvasir/{phase}.txt').write_text('\n'.join(self.ids) + '\n')
            report = Path(tmp) / 'report.json'
            result = subprocess.run([sys.executable, str(source / 'check_inputs.py'),
                                     '--dataset', str(self.dataset), '--workers', '2',
                                     '--output', str(report)], capture_output=True, text=True,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', NO_ALBUMENTATIONS_UPDATE='1'),
                                    timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = json.loads(report.read_text())
            self.assertEqual(saved['status'], 'passed')
            self.assertEqual(len(saved['captures']), 4)
            self.assertEqual(saved['captures'][0]['records'], saved['captures'][2]['records'])

    def test_stable_seed_does_not_depend_on_python_hash_seed(self):
        outputs = []
        for hash_seed in ('1', '1234'):
            outputs.append(subprocess.check_output([sys.executable, '-c',
                'from paired_inputs import stable_seed; print(stable_seed(1234,"augmentation",0,"a.png"))'],
                cwd=self.source, text=True, env=dict(os.environ, PYTHONHASHSEED=hash_seed,
                                                   NO_ALBUMENTATIONS_UPDATE='1')))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
