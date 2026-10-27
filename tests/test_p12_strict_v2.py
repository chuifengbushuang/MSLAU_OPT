"""Original experiment3 on the verified strict-v2 baseline; no new NAF design."""
import ast
import contextlib
import hashlib
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

from test_exp07_naf import definitions, load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_experiment import build, verify_base

EXPERIMENT = 'exp03_naf_p12_paired_strict_v2'
BASELINE = 'exp00_baseline_paired_strict_v2'

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None

if os.environ.get('P12_STRICT_FULL_SMOKE') == '1' and torch is None:
    raise ImportError('Requested strict P1/P2 Hiera check requires PyTorch')


@unittest.skipIf(torch is None, 'PyTorch required')
class P12StrictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = Path(cls.temp.name) / 'source'
        cls.base = Path(cls.temp.name) / 'baseline'
        build(EXPERIMENT, cls.source)
        build(BASELINE, cls.base)
        cls.naf = load('p12_strict_module', cls.source / 'networks/naf_select_reg.py')
        names = ['BasicConv2d', 'RFBModified', 'SAM2UNetUp', 'SAM2UNetDecoder']
        cls.decoder = definitions(cls.source / 'networks/mslau_net.py', names,
            dict(torch=torch, nn=nn, F=F, PyramidNAF2D=cls.naf.PyramidNAF2D))['SAM2UNetDecoder']
        cls.original = definitions(cls.base / 'networks/mslau_net.py', names,
            dict(torch=torch, nn=nn, F=F))['SAM2UNetDecoder']
        cls.helpers = definitions(cls.source / 'paired_inputs.py',
            ['stable_seed', 'set_loader_epoch', 'InputDigest', 'update_tensor_digest',
             'state_digest', 'append_record', 'record_initial_state'],
            dict(torch=torch, np=np, random=random, json=json, Path=Path, hashlib=hashlib,
                 PROTOCOL='paired_aug_v2', SEED_NAMESPACE='paired_aug_v1'))
        cls.runtime = definitions(cls.source / 'strict_runtime.py',
            ['configure_runtime', 'runtime_record', 'record_setup', 'record_phase_state', 'record_selected_states'],
            dict(torch=torch, os=os, json=json, Path=Path, hashlib=hashlib, PROTOCOL='strict_cuda_v1',
                 append_record=cls.helpers['append_record'], state_digest=cls.helpers['state_digest']))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.flags = (torch.are_deterministic_algorithms_enabled(),
                      torch.is_deterministic_algorithms_warn_only_enabled(),
                      torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.flags[0], warn_only=self.flags[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = self.flags[2:]

    def test_preserves_original_p12_model_and_matches_v2_inputs_and_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / 'original'
            build('exp03_naf_p12', original)
            self.assertEqual((self.source / 'networks/mslau_net.py').read_bytes(),
                             (original / 'networks/mslau_net.py').read_bytes())
        self.assertEqual((self.source / 'networks/naf_select_reg.py').read_bytes(),
            (ROOT / 'experiments/shared_naf_v1/added/networks/naf_select_reg.py').read_bytes())
        for name in ('paired_inputs.py', 'strict_runtime.py'):
            self.assertEqual((self.source / name).read_bytes(),
                (ROOT / 'experiments/exp09_naf_discrete_paired_strict_v2/added' / name).read_bytes())
        a = json.loads((self.source / 'config.json').read_text())
        b = json.loads((self.base / 'config.json').read_text())
        self.assertEqual(a['train_options'].pop('naf_stages'), '1,2')
        self.assertEqual(a['eval_options'].pop('naf_stages'), '1,2')
        self.assertEqual(a['train_options'].pop('naf_reg_weight'), .001)
        a['id'] = b['id']
        self.assertEqual(a, b)
        self.assertEqual((self.source / 'run.py').read_bytes(), (self.base / 'run.py').read_bytes())
        for name in ('get_train_transform', 'get_valid_transform', 'load_ids'):
            nodes = [next(n for n in ast.parse((p / 'kvasir_train.py').read_text()).body
                          if isinstance(n, ast.FunctionDef) and n.name == name) for p in (self.source, self.base)]
            self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]))
        for path in self.source.rglob('*.py'):
            ast.parse(path.read_text())
        launcher = load('p12_strict_launcher', self.source / 'run.py')
        train, evaluate = launcher.commands(1234, 'preview')
        for command in (train, evaluate):
            self.assertEqual(command[command.index('--naf_stages') + 1], '1,2')
        verify_base()

    def test_three_seeds_preserve_shared_initialization_rng_and_baseline_hash(self):
        for seed in (1234, 42, 2026):
            torch.manual_seed(seed)
            reference = self.original((4, 8, 16, 32), 4)
            rng = torch.get_rng_state()
            torch.manual_seed(seed)
            decoder = self.decoder((4, 8, 16, 32), 4, naf_stages=(1, 2))
            torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
            self.assertEqual(decoder.frequency_enhancer.active_stages, (1, 2))
            self.assertEqual(sum(p.numel() for p in decoder.frequency_enhancer.parameters()), 164)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(value, decoder.state_dict()[name], atol=0, rtol=0)
            wrapper, baseline = nn.Module(), nn.Module()
            wrapper.progressive_decoder = decoder
            baseline.progressive_decoder = reference
            with tempfile.TemporaryDirectory() as tmp:
                record = self.helpers['record_initial_state'](wrapper, seed, tmp)
            self.assertEqual(record['baseline_state_sha256'], self.helpers['state_digest'](baseline.state_dict()))

    def test_actual_p12_training_repeats_and_preserves_three_checkpoints(self):
        decoder_type = self.decoder
        class Model(nn.Module):
            decoder_mode, conv_mla, naf_reg_weight = 'p5_sam2unet', None, .001
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Conv2d(3, c, 1) for c in (4, 8, 16, 32)])
                self.progressive_decoder = decoder_type((4, 8, 16, 32), 4, naf_stages=(1, 2))
            def forward(self, x, **kwargs):
                features = [layer(F.interpolate(x, size=(h, h))) for layer, h in zip(self.layers, (32, 16, 8, 4))]
                return self.progressive_decoder(features, 32, True)
        class Dataset(torch.utils.data.Dataset):
            seed, epoch = 1234, 0
            def __init__(self, phase): self.phase = phase
            def __len__(self): return 2
            def __getitem__(self, index):
                x = torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(index+11*self.epoch))
                return x, (x[0] > 0).float(), str(index)
        ns = {**self.helpers, **self.runtime, 'torch': torch, 'nn': nn, 'F': F,
              'Variable': torch.autograd.Variable, 'os': os, 'logging': logging, 'time': time,
              'STATS_NAMES': self.naf.STATS_NAMES}
        definitions(self.source / 'kvasir_train.py',
            ['train_model', 'snapshot_state_dict', 'atomic_save_state_dict', 'log_fusion_scale_stats'], ns)
        criterion = definitions(self.source / 'loss.py', ['StructureLoss_binary'],
            dict(torch=torch, nn=nn, F=F))['StructureLoss_binary']()
        captured = []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}), \
             patch('torch.cuda.is_available', return_value=False):
            self.runtime['configure_runtime']()
            for repeat in (1, 2):
                torch.manual_seed(1234)
                model = Model()
                output = Path(tmp) / str(repeat)
                logs = output / 'logs'
                logs.mkdir(parents=True)
                self.helpers['record_initial_state'](model, 1234, logs)
                self.runtime['record_setup'](model, 1234, logs)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
                loaders = {phase: torch.utils.data.DataLoader(Dataset(phase), batch_size=2,
                           generator=torch.Generator()) for phase in ('train', 'valid')}
                with contextlib.redirect_stdout(io.StringIO()):
                    ns['train_model'](model, criterion, optimizer, scheduler, loaders,
                        lambda x,y: (x.sigmoid()-y.unsqueeze(1)).abs().mean(), 2, str(output),
                        aux_d2_weight=1, aux_d3_weight=1, boundary_weight=0,
                        input_audit_path=str(logs / 'input_fingerprints.jsonl'))
                self.assertEqual(len(list(output.glob('*.pth'))), 3)
                captured.append({p.name:p.read_text() for p in logs.iterdir()})
                for p in output.glob('*.pth'):
                    key = 'best_loss_sha256' if p.name.startswith('best_loss_') else 'best_iou_sha256'
                    self.assertEqual(self.helpers['state_digest'](torch.load(p, weights_only=True)),
                                     json.loads((logs / 'checkpoint_tensors.json').read_text())[key])
            self.assertEqual(*captured)
            self.assertEqual(len(captured[0]['numerical_states.jsonl'].splitlines()), 4)

    @unittest.skipUnless(os.environ.get('P12_STRICT_FULL_SMOKE') == '1', 'server Hiera synthetic check not requested')
    def test_full_hiera_strict_two_synthetic_updates(self):
        checkpoint = os.environ.get('HIERA_CHECKPOINT',
            '/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors')
        self.assertTrue(Path(checkpoint).is_file(), checkpoint)
        self.runtime['configure_runtime']()
        device = torch.device('cuda')
        batch = int(os.environ.get('P12_STRICT_SMOKE_BATCH', '12'))
        previous = {name:sys.modules.pop(name) for name in list(sys.modules)
                    if name == 'networks' or name.startswith('networks.')}
        sys.path.insert(0, str(self.source))
        try:
            from networks.mslau_net import MSLAU_net
            torch.manual_seed(1234)
            model = MSLAU_net(img_size=352, decoder_mode='p5_sam2unet', encoder_name='sam2_hiera_large',
                hiera_checkpoint=checkpoint, naf_stages=(1, 2)).to(device).train()
            self.assertTrue(model.encoder.freeze_backbone)
            enhancer = model.progressive_decoder.frequency_enhancer
            self.assertEqual(enhancer.active_stages, (1, 2))
            self.assertEqual(sum(p.numel() for p in enhancer.parameters()), 164)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=5e-4)
            criterion = definitions(self.source / 'loss.py', ['StructureLoss_binary'],
                dict(torch=torch, nn=nn, F=F))['StructureLoss_binary']()
            images = torch.randn(batch, 3, 352, 352, device=device)
            target = torch.randint(0, 2, (batch, 1, 352, 352), device=device).float()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                output = model(images, return_aux=True)
                task = sum(criterion(output[k], target) for k in ('logits', 'aux_d2', 'aux_d3'))
                objective = task + .001 * output['naf_regularization']
                objective.backward()
                self.assertTrue(torch.isfinite(objective))
                for name,p in enhancer.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                self.assertTrue(torch.isfinite(model(images)).all())
            print('P12_STRICT_FULL_SMOKE:', json.dumps(dict(batch=batch, added_parameters=164,
                stages=[1,2], strict=True, synthetic_updates=2)), flush=True)
        finally:
            sys.path.remove(str(self.source))
            for name in list(sys.modules):
                if name == 'networks' or name.startswith('networks.'):
                    del sys.modules[name]
            sys.modules.update(previous)


if __name__ == '__main__':
    unittest.main()
