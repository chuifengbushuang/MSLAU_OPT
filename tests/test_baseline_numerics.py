"""Diagnostic fidelity and first-divergence detection; no real Hiera training."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import check_baseline_numerics as probe

try:
    import numpy as np
    import torch
except ImportError:
    torch = None


def helpers():
    return probe.definitions(ROOT / 'experiments/exp00_baseline_paired_aug_v1/added/paired_inputs.py',
        {'state_digest', 'update_tensor_digest'}, dict(torch=torch, json=json, hashlib=hashlib))


def fixture(lr=1e-3, nan=False):
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
            self.bn = torch.nn.BatchNorm2d(4)
            self.heads = torch.nn.ModuleList([torch.nn.Conv2d(4, 1, 1) for _ in range(3)])

        def forward(self, images, return_aux=False):
            assert return_aux
            features = self.bn(self.conv(images)).relu()
            return {key: torch.nn.functional.interpolate(head(features), size=(12, 12),
                        mode='bilinear', align_corners=key == 'logits')
                    for key, head in zip(('logits', 'aux_d2', 'aux_d3'), self.heads)}

    model = Model()
    criterion = probe.definitions(ROOT / 'experiments/exp00_baseline/loss.py',
        {'StructureLoss_binary'}, dict(torch=torch, nn=torch.nn, F=torch.nn.functional))['StructureLoss_binary']()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    batches = [(torch.randn(2, 3, 6, 6), torch.randint(0, 2, (2, 12, 12)).float(), ['a', 'b'])
               for _ in range(3)]
    if nan:
        batches[0][0].fill_(float('nan'))
    digest = helpers()['state_digest']
    initial = digest(model.state_dict())
    with contextlib.redirect_stdout(io.StringIO()):
        records = probe.replay_steps(model, optimizer, criterion, batches, digest,
                                     torch.device('cpu'), (1.0, 1.0))
    return dict(initial=initial, steps=records)


class SourceTests(unittest.TestCase):
    def test_worker_failure_is_preserved_without_launching_a_second_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'records'
            result = subprocess.run([sys.executable, str(ROOT / 'tools/check_baseline_numerics.py'),
                '--output', str(output), '--dataset', str(root / 'absent_data')],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(result.returncode, 1, result.stdout)
            summary = json.loads((output / 'summary.json').read_text())
            report = json.loads((output / 'replay_1.json').read_text())
            self.assertEqual(summary['status'], 'failed')
            self.assertEqual(summary['failed_replay'], 1)
            self.assertIn('Dataset directory not found', report['error'])
            self.assertFalse((output / 'replay_2.json').exists())
            self.assertEqual({p.suffix for p in output.iterdir()}, {'.json', '.log'})

    def test_definition_extraction_avoids_unrelated_import_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.py'
            path.write_text('raise RuntimeError("must not import")\ndef wanted():\n    return 17\n')
            self.assertEqual(probe.definitions(path, {'wanted'}, {})['wanted'](), 17)
            with self.assertRaises(ValueError):
                probe.definitions(path, {'absent'}, {})

    def test_differences_retain_stage_and_parameter_name(self):
        a = {'initial': 'same', 'steps': [{'forward': {'head': 'same'},
                                         'backward': {'weight': 'a'}}]}
        b = {'initial': 'same', 'steps': [{'forward': {'head': 'same'},
                                         'backward': {'weight': 'b'}}]}
        self.assertEqual(probe.first_difference(a, b), 'steps[0].backward.weight')
        self.assertIsNone(probe.first_difference(a, a))


@unittest.skipIf(torch is None, 'PyTorch required')
class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_adamw_updates_and_bn_buffers_are_recorded_and_repeat(self):
        a, b = fixture(), fixture()
        self.assertIsNone(probe.first_difference(a, b))
        self.assertEqual(len(a['steps']), 3)
        first = a['steps'][0]
        losses = first['forward']['losses']
        self.assertEqual(losses['total'],
                         (torch.tensor(losses['logits']) + torch.tensor(losses['aux_d2']) +
                          torch.tensor(losses['aux_d3'])).item())
        self.assertGreater(first['backward']['norm'], 0)
        self.assertEqual(first['backward']['missing'], [])
        self.assertIn('bn.running_mean', first['update']['buffers'])
        self.assertNotEqual(first['update']['parameters'], a['steps'][1]['update']['parameters'])

    def test_changed_lr_is_first_detected_at_update_not_input_or_gradient(self):
        difference = probe.first_difference(fixture(), fixture(lr=2e-3))
        self.assertTrue(difference.startswith('steps[0].update.parameters.'), difference)

    def test_nonfinite_computation_cannot_report_a_reproducibility_pass(self):
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite loss at step 0'):
            fixture(nan=True)

    def test_fresh_processes_match_without_saving_any_tensors(self):
        script = ('import json,sys,torch; '
                  f'sys.path.insert(0, {str(ROOT / "tests")!r}); '
                  'from test_baseline_numerics import fixture; torch.set_num_threads(1); '
                  'print(json.dumps(fixture()))')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONHASHSEED='1234', PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory() as tmp:
            records = [json.loads(subprocess.check_output([sys.executable, '-c', script],
                       cwd=tmp, env=env, text=True)) for _ in range(2)]
            self.assertIsNone(probe.first_difference(*records))
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
