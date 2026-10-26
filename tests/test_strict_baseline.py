"""Strict baseline isolation and actual small CPU train/validation cycles."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_experiment import build, verify_base
from check_baseline_numerics import definitions

EXPERIMENT = 'exp00_baseline_paired_strict_v1'

try:
    import numpy as np
    import torch
except ImportError:
    torch = None


class DefinitionTests(unittest.TestCase):
    def test_launcher_sets_environment_and_rejects_non_strict_audits(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'source'
                build(EXPERIMENT, source)
                spec = importlib.util.spec_from_file_location('strict_launcher', source / 'run.py')
                launcher = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(launcher)
                calls = []

                class Child:
                    def __init__(self, command, **kwargs):
                        calls.append(kwargs['env'])
                        if '--output_dir' in command:
                            checkpoints = Path(command[command.index('--output_dir') + 1])
                            for name in ('kvasir_best_model.pth', 'best_iou_named.pth', 'best_loss_named.pth'):
                                (checkpoints / name).write_bytes(b'fixture')
                            logs = Path(command[command.index('--log_dir') + 1])
                            logs.mkdir()
                            (logs / 'initial_state.json').write_text('{}')
                            rows = [dict(epoch=e, phase=phase, deterministic=enabled, warn_only=False)
                                    for e in range(20) for phase in ('train', 'valid')]
                            for name in ('input_fingerprints.jsonl', 'epoch_metrics.jsonl', 'numerical_states.jsonl'):
                                (logs / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
                            (logs / 'numerical_runtime.json').write_text(json.dumps(rows[0]))
                            (logs / 'checkpoint_tensors.json').write_text('{"best_iou_sha256":"fixture"}')
                            self.stdout = ['Best val IoU: 0.87 at epoch 19\n']
                        else:
                            self.stdout = ['mean IoU: 0.87\n']
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                    def wait(self):
                        return 0

                with patch.object(launcher, 'OUTPUT', root / 'output'), \
                     patch.object(launcher, 'DATASET', str(root)), \
                     patch.object(launcher, 'HIERA', __file__), \
                     patch.object(launcher.subprocess, 'check_output', return_value='fixture'), \
                     patch.object(launcher.subprocess, 'Popen', side_effect=Child), \
                     patch.dict(os.environ, {'P51_REPEAT': '2', 'CUBLAS_WORKSPACE_CONFIG': 'wrong'}), \
                     patch('sys.argv', ['run.py', '--gpu', '3', '--run']), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if enabled:
                        launcher.main()
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'non-strict numerical audit'):
                            launcher.main()
                manifest = json.loads(next((root / 'output/experiments').glob('*/manifest.json')).read_text())
                self.assertEqual(manifest['status'], 'completed' if enabled else 'failed')
                self.assertTrue(manifest['run_id'].endswith('_r2'))
                self.assertEqual(manifest['numerical_protocol'], 'strict_cuda_v1')
                self.assertEqual(len(manifest['checkpoint_files']), 3)
                self.assertEqual(len(calls), 2)
                for environment in calls:
                    self.assertEqual(environment['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
                    self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '3')
                    self.assertEqual(environment['PYTHONHASHSEED'], '1234')

    def test_only_numerical_protocol_and_audits_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, strict = (Path(tmp) / name for name in ('paired', 'strict'))
            build('exp00_baseline_paired_aug_v1', original)
            build(EXPERIMENT, strict)
            a, b = [json.loads((p / 'config.json').read_text()) for p in (original, strict)]
            b['id'] = a['id']
            self.assertEqual(b.pop('numerical_protocol'), 'strict_cuda_v1')
            self.assertEqual(a, b)
            for name in verify_base()['files']:
                if name not in {'kvasir_train.py', 'kvasir_test.py'}:
                    self.assertEqual((original / name).read_bytes(), (strict / name).read_bytes())
            self.assertEqual((original / 'paired_inputs.py').read_bytes(), (strict / 'paired_inputs.py').read_bytes())
            for name in ('get_train_transform', 'get_valid_transform', 'load_ids'):
                nodes = [next(node for node in ast.parse((p / 'kvasir_train.py').read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == name) for p in (original, strict)]
                self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]))
            for p in strict.rglob('*.py'):
                ast.parse(p.read_text())
            train = (strict / 'kvasir_train.py').read_text()
            evaluation = (strict / 'kvasir_test.py').read_text()
            launcher = (strict / 'run.py').read_text()
            self.assertIn('configure_runtime()', train)
            self.assertIn('configure_runtime()', evaluation)
            self.assertIn('CUBLAS_WORKSPACE_CONFIG=":4096:8"', launcher)
            self.assertNotIn('warn_only=True', train + evaluation)


@unittest.skipIf(torch is None, 'PyTorch required')
class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = Path(cls.temp.name) / 'source'
        build(EXPERIMENT, cls.source)
        cls.data = definitions(cls.source / 'paired_inputs.py',
            {'stable_seed', 'set_loader_epoch', 'update_tensor_digest', 'state_digest',
             'InputDigest', 'append_record', 'record_initial_state'},
            dict(torch=torch, np=np, random=random, json=json, hashlib=hashlib, Path=Path, PROTOCOL='paired_aug_v1'))
        cls.runtime = definitions(cls.source / 'strict_runtime.py',
            {'configure_runtime', 'runtime_record', 'record_setup', 'record_phase_state', 'record_selected_states'},
            dict(torch=torch, hashlib=hashlib, json=json, os=os, Path=Path, PROTOCOL='strict_cuda_v1', **{
                key: cls.data[key] for key in ('append_record', 'state_digest')}))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.previous = (torch.are_deterministic_algorithms_enabled(),
                         torch.is_deterministic_algorithms_warn_only_enabled(),
                         torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous[0], warn_only=self.previous[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = self.previous[2:]

    def test_strict_setup_requires_workspace_and_preserves_precision_flags(self):
        precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ''}):
            with self.assertRaisesRegex(RuntimeError, 'before Python starts'):
                self.runtime['configure_runtime']()
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            self.runtime['configure_runtime']()
            record = self.runtime['runtime_record']()
        self.assertTrue(record['deterministic'])
        self.assertFalse(record['warn_only'])
        self.assertEqual(precision, (record['matmul_tf32'], record['cudnn_tf32']))

    def run_training(self, directory):
        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)
        directory.mkdir()
        logs = directory / 'logs'
        logs.mkdir()

        class Dataset(torch.utils.data.Dataset):
            seed, epoch = 1234, 0
            def __init__(self, phase):
                self.phase = phase
            def __len__(self):
                return 6
            def __getitem__(self, index):
                generator = torch.Generator().manual_seed(index + self.epoch * 11)
                image = torch.randn(3, 8, 8, generator=generator)
                return image, (image[0] > 0).float(), str(index)

        class Model(torch.nn.Module):
            decoder_mode, conv_mla = 'p5_sam2unet', None
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Conv2d(3, 4, 1)
                self.bn = torch.nn.BatchNorm2d(4)
                self.heads = torch.nn.ModuleList([torch.nn.Conv2d(4, 1, 1) for _ in range(3)])
            def forward(self, images, **kwargs):
                features = self.bn(self.encoder(images)).relu()
                return {key: head(features) for key, head in
                        zip(('logits', 'aux_d2', 'aux_d3'), self.heads)}

        model = Model()
        loaders = {phase: torch.utils.data.DataLoader(Dataset(phase), batch_size=2,
                   generator=torch.Generator()) for phase in ('train', 'valid')}
        self.data['record_initial_state'](model, 1234, logs)
        self.runtime['record_setup'](model, 1234, logs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
        names = {'train_model', 'snapshot_state_dict', 'atomic_save_state_dict', 'log_fusion_scale_stats'}
        namespace = definitions(self.source / 'kvasir_train.py', names, dict(torch=torch,
            Variable=torch.autograd.Variable, os=os, logging=logging, time=time,
            **{key: self.data[key] for key in ('InputDigest', 'set_loader_epoch', 'append_record')},
            **{key: self.runtime[key] for key in ('record_phase_state', 'record_selected_states')}))
        criterion = definitions(self.source / 'loss.py', {'StructureLoss_binary'},
            dict(torch=torch, nn=torch.nn, F=torch.nn.functional))['StructureLoss_binary']()
        with contextlib.redirect_stdout(io.StringIO()):
            namespace['train_model'](model, criterion, optimizer, scheduler, loaders,
                lambda logits, target: (logits.sigmoid() - target.unsqueeze(1)).abs().mean(),
                3, str(directory), aux_d2_weight=1, aux_d3_weight=1, boundary_weight=0,
                input_audit_path=str(logs / 'input_fingerprints.jsonl'))
        return {p.name: p.read_text() for p in logs.iterdir()}

    def test_real_training_validation_scheduler_audits_and_three_checkpoints_repeat(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}), \
             patch('torch.cuda.is_available', return_value=False):
            self.runtime['configure_runtime']()
            root = Path(tmp)
            a, b = [self.run_training(root / name) for name in ('a', 'b')]
            self.assertEqual(a, b)
            rows = [json.loads(line) for line in a['numerical_states.jsonl'].splitlines()]
            self.assertEqual([(r['epoch'], r['phase']) for r in rows],
                             [(e, phase) for e in range(3) for phase in ('train', 'valid')])
            self.assertEqual(rows[0]['parameters_sha256'], rows[1]['parameters_sha256'])
            self.assertEqual(rows[0]['optimizer_sha256'], rows[1]['optimizer_sha256'])
            self.assertEqual(rows[0]['buffers_sha256'], rows[1]['buffers_sha256'])
            self.assertNotEqual(rows[0]['parameters_sha256'], rows[2]['parameters_sha256'])
            self.assertNotEqual(rows[0]['learning_rates'], rows[1]['learning_rates'])
            self.assertTrue(all(r['deterministic'] and not r['warn_only'] for r in rows))
            checkpoints = list((root / 'a').glob('*.pth'))
            self.assertEqual(len(checkpoints), 3)
            digests = json.loads(a['checkpoint_tensors.json'])
            for path in checkpoints:
                kind = 'best_loss' if path.name.startswith('best_loss_') else 'best_iou'
                self.assertEqual(self.data['state_digest'](torch.load(path, weights_only=True)), digests[kind + '_sha256'])


if __name__ == '__main__':
    unittest.main()
