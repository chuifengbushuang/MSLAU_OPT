"""Discrete-mask numerics, matched inputs and actual rendered-source integration.

EXP09_FULL_SMOKE=1 EXP09_DEVICE=cuda EXP09_SMOKE_BATCH=12 enables a real
pretrained Hiera synthetic forward/backward check; never real-data training.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from test_exp07_naf import definitions, load

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None

try:
    import albumentations as A
    import numpy as np
    from albumentations.pytorch import ToTensorV2
    from PIL import Image
except ImportError:
    A = None

if os.environ.get('EXP09_FULL_SMOKE') == '1' and torch is None:
    raise ImportError('Requested experiment-9 Hiera check requires PyTorch')

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/exp00_baseline'
EXPERIMENT = 'exp09_naf_discrete'


@unittest.skipIf(torch is None, 'PyTorch required')
class DiscreteNAFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix='exp09_test_')
        cls.source = Path(cls.temporary.name) / 'exp09'
        builder = load('exp09_builder', ROOT / 'tools/build_experiment.py')
        cls.manifest = builder.build(EXPERIMENT, cls.source)
        cls.naf = load('exp09_module', cls.source / 'networks/naf_discrete.py')
        cls.reference = load('exp09_reference', ROOT / 'experiments/exp07_naf_cascade/added/networks/naf_cascade.py')
        names = ['BasicConv2d', 'RFBModified', 'SAM2UNetUp', 'SAM2UNetDecoder']
        cls.original = definitions(BASE / 'networks/mslau_net.py', names, dict(torch=torch, nn=nn, F=F))
        cls.rendered = definitions(cls.source / 'networks/mslau_net.py', names,
            dict(torch=torch, nn=nn, F=F, CascadeDiscreteNAF2D=cls.naf.CascadeDiscreteNAF2D))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def block(self, level=2, channels=4):
        block = self.naf.DualInputDiscreteNAF2D(level, channels)
        block.set_selection_context(1234, 0, ['a', 'b'])
        return block

    def test_standalone_config_and_frozen_base(self):
        baseline = json.loads((BASE / 'config.json').read_text())
        config = json.loads((self.source / 'config.json').read_text())
        options = config['train_options'].copy()
        self.assertEqual(options.pop('naf_reg_weight'), .001)
        self.assertEqual(options, baseline['train_options'])
        self.assertEqual(config['eval_options'], baseline['eval_options'])
        self.assertEqual(config['input_protocol'], 'paired_aug_v1')
        self.assertEqual(config['selection_protocol'], 'naf_discrete_v1')
        self.assertEqual(set(self.manifest['added_files']), {'paired_inputs.py', 'networks/naf_discrete.py'})
        frozen = json.loads((BASE / 'source_manifest.json').read_text())
        for name, expected in frozen['files'].items():
            self.assertEqual(hashlib.sha256((BASE / name).read_bytes()).hexdigest(), expected)
        for name in ('SAM2UNetUp', 'RFBModified', 'BasicConv2d'):
            old = next(n for n in ast.parse((BASE / 'networks/mslau_net.py').read_text()).body
                       if isinstance(n, ast.ClassDef) and n.name == name)
            new = next(n for n in ast.parse((self.source / 'networks/mslau_net.py').read_text()).body
                       if isinstance(n, ast.ClassDef) and n.name == name)
            self.assertEqual(ast.dump(old), ast.dump(new))
        for path in self.source.rglob('*.py'):
            ast.parse(path.read_text())
        launcher = load('exp09_launcher', self.source / 'run.py')
        train, evaluate = launcher.commands(1234, 'preview')
        self.assertIn('--naf_reg_weight', train)
        self.assertNotIn('--naf_reg_weight', evaluate)
        self.assertTrue(evaluate[evaluate.index('--model') + 1].endswith('/kvasir_best_model.pth'))
        for text in ((self.source / 'kvasir_train.py').read_text(),
                     (self.source / 'networks/mslau_net.py').read_text()):
            self.assertNotIn('networks.naf_cascade', text)
            self.assertNotIn('shared_naf', text)

    def test_exact_pair_budgets_and_real_reconstruction_for_odd_even_grids(self):
        for h, w in ((11, 11), (22, 22), (44, 44), (88, 88), (13, 18), (12, 17)):
            block = self.block()
            inputs = [torch.randn(2, 4, h, w) for _ in range(2)]
            spectra = [torch.fft.rfft2(t, norm='ortho') for t in inputs]
            gates, masks, values, _, bands, anchors, _ = block.selection(*spectra, (h, w))
            expected_anchors = (1 + (h % 2 == 0)) * (1 + (w % 2 == 0))
            self.assertEqual(anchors, expected_anchors)
            self.assertEqual(masks.shape[-1], (h * w - anchors) // 2)
            self.assertTrue(((masks == 0) | (masks == 1)).all())
            torch.testing.assert_close(masks.sum(-1), masks.new_full((2, 2), math.ceil(.9 * masks.shape[-1])))
            self.assertTrue((gates[..., 0, 0] == 1).all())
            for b in range(2):
                target = spectra[b] * gates[:, b:b + 1]
                restored = torch.fft.irfft2(target, s=(h, w), norm='ortho')
                torch.testing.assert_close(torch.fft.rfft2(restored, norm='ortho'), target,
                                           atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(restored.mean((-2, -1)), inputs[b].mean((-2, -1)),
                                           atol=1e-6, rtol=1e-5)
                self.assertLessEqual(restored.square().sum().item(), inputs[b].square().sum().item() + .01)
                # A removed coefficient is actually zero, not 90% restored by alpha.
                dropped = gates[:, b:b + 1] == 0
                self.assertTrue(dropped.any())
                self.assertEqual(target.masked_select(dropped).abs().max().item(), 0)
            filtered = block.filter_pair(*inputs)
            for actual, q, gate in zip(filtered[:2], spectra, gates.unbind(1)):
                torch.testing.assert_close(actual, torch.fft.irfft2(q * gate[:, None], s=(h, w), norm='ortho'))
            constant = torch.full_like(inputs[0], 3)
            for actual in block.filter_pair(constant, constant)[:2]:
                torch.testing.assert_close(actual, constant)

    def test_soft_surrogate_implicit_gradient_and_hard_forward(self):
        scores = torch.randn(2, 13, dtype=torch.double, requires_grad=True)
        relaxed = self.naf._BudgetedSigmoid.apply(scores, 9, .3)
        torch.testing.assert_close(relaxed.sum(-1), torch.full((2,), 9., dtype=torch.double))
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: self.naf._BudgetedSigmoid.apply(x, 9, .3), (scores,), atol=1e-5, rtol=1e-4))
        hard = self.naf.discrete_mask(scores, 9, True, torch.zeros_like(scores), tau=.3)
        self.assertTrue(((hard == 0) | (hard == 1)).all())
        weights = torch.arange(13, dtype=torch.double)
        gradient = torch.autograd.grad((hard * weights).sum(), scores)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().max().item(), 0)
        torch.testing.assert_close(gradient.sum(-1), torch.zeros(2, dtype=torch.double), atol=1e-12, rtol=0)
        ties = torch.zeros(1, 8)
        expected = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=ties.dtype)
        torch.testing.assert_close(self.naf.discrete_mask(ties, 5, False), expected)

    def test_sampling_key_is_independent_of_batch_order_repeat_and_global_rng(self):
        a = self.naf.sample_gumbels(1234, 0, ['a', 'b'], 3, 'local', 240, 'cpu', torch.float32)
        torch.rand(39)
        state = torch.get_rng_state()
        b = self.naf.sample_gumbels(1234, 0, ['b', 'a'], 3, 'local', 240, 'cpu', torch.float32)
        torch.testing.assert_close(a, b.flip(0), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        for seed, epoch, branch in ((42, 0, 'local'), (1234, 1, 'local'), (1234, 0, 'context')):
            other = self.naf.sample_gumbels(seed, epoch, ['a', 'b'], 3, branch, 240, 'cpu', torch.float32)
            self.assertFalse(torch.equal(a, other))
        block = self.block()
        x = torch.randn(2, 4, 22, 22)
        q = torch.fft.rfft2(x, norm='ortho')
        first = block.selection(q, q, (22, 22))[0]
        torch.rand(20)
        torch.testing.assert_close(first, block.selection(q, q, (22, 22))[0], rtol=0, atol=0)
        block.set_selection_context(1234, 1, ['a', 'b'])
        self.assertFalse(torch.equal(first, block.selection(q, q, (22, 22))[0]))
        block.eval()
        first = block.selection(q, q, (22, 22))[0]
        block.set_selection_context(42, 8, ['other', 'ids'])
        torch.testing.assert_close(first, block.selection(q, q, (22, 22))[0], rtol=0, atol=0)
        block.train()
        block.selection_context = None
        with self.assertRaisesRegex(RuntimeError, 'seed/epoch/sample'):
            block(x, x)

    def test_regularizer_is_score_squared_not_hard_mask_and_uses_six_means(self):
        block = self.block()
        nn.init.normal_(block.selector[-1].weight, std=.2)
        x, c = torch.randn(2, 4, 22, 22), torch.randn(2, 4, 22, 22)
        spectra = [torch.fft.rfft2(t, norm='ortho') for t in (x, c)]
        _, first_mask, values, penalties, bands, _, _ = block.selection(*spectra, (22, 22))
        for b, n in enumerate((2, 3)):
            expected = (values[:, b].square() * n ** (-(bands + 1).to(values.dtype))).mean()
            torch.testing.assert_close(penalties[b], expected)
        block.set_selection_context(1234, 1, ['a', 'b'])
        _, second_mask, _, repeated, _, _, _ = block.selection(*spectra, (22, 22))
        self.assertFalse(torch.equal(first_mask, second_mask))
        torch.testing.assert_close(penalties, repeated, atol=0, rtol=0)
        penalties.sum().backward()
        self.assertGreater(block.selector[-1].weight.grad.abs().max().item(), 0)
        cascade = self.naf.CascadeDiscreteNAF2D(4).eval()
        features = [torch.randn(2, 4, h, h, requires_grad=True) for h in (32, 16, 8, 4)]
        seen, order, handles = {}, [], []
        for i in (1, 2, 3):
            module = cascade.stages['p%d' % i]
            nn.init.normal_(module.output_projection.weight, std=.1)
            def capture(module, inputs, outputs, level=i):
                seen[level] = (inputs, outputs)
                order.append(level)
            handles.append(module.register_forward_hook(capture))
        try:
            enhanced, regularizer, stats, fusion = cascade(features)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(order, [3, 2, 1])
        self.assertIs(seen[3][0][1], features[3])
        self.assertIs(seen[2][0][1], seen[3][1][0])
        self.assertIs(seen[1][0][1], seen[2][1][0])
        self.assertIs(enhanced[3], features[3])
        torch.testing.assert_close(regularizer, sum(seen[i][1][1].sum() for i in (1, 2, 3)) / 6)
        self.assertEqual(stats.shape, (3, 2, len(self.naf.FILTER_STATS_NAMES)))
        self.assertEqual(fusion.shape, (3, len(self.naf.FUSION_STATS_NAMES)))
        enhanced[0].square().mean().backward()
        self.assertGreater(cascade.stages['p3'].output_projection.weight.grad.abs().max().item(), 0)

    def test_shared_initialization_and_original_decoder_outputs_match(self):
        for seed in (1234, 42, 2026):
            torch.manual_seed(seed)
            ref = self.reference.CascadeNAF2D(64)
            state = torch.get_rng_state()
            new = self.naf.CascadeDiscreteNAF2D(64)
            torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
            self.assertEqual(sum(p.numel() for p in new.parameters()), 56118)
            self.assertFalse(any('alpha' in name for name, _ in new.named_parameters()))
            for name, value in new.state_dict().items():
                torch.testing.assert_close(value, ref.state_dict()[name], atol=0, rtol=0)
            torch.manual_seed(seed)
            original = self.original['SAM2UNetDecoder']((4, 8, 16, 32), 4)
            original_rng = torch.get_rng_state()
            torch.manual_seed(seed)
            actual = self.rendered['SAM2UNetDecoder']((4, 8, 16, 32), 4)
            torch.testing.assert_close(torch.get_rng_state(), original_rng, atol=0, rtol=0)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, actual.state_dict()[name], atol=0, rtol=0)
            features = [torch.randn(2, c, h, h) for c, h in zip((4, 8, 16, 32), (32, 16, 8, 4))]
            original.eval(); actual.eval()
            with torch.no_grad():
                a, b = original(features, 64, True), actual(features, 64, True)
            for name in ('logits', 'aux_d2', 'aux_d3'):
                torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)

    def test_selector_receives_task_gradient_after_output_projection_opens(self):
        block = self.block()
        optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
        x, c = torch.randn(2, 4, 22, 22), torch.randn(2, 4, 11, 11)
        output, penalties, *_ = block(x, c)
        torch.testing.assert_close(output, x, atol=0, rtol=0)
        (output.square().mean() + .001 * penalties.sum()).backward()
        self.assertGreater(block.output_projection.weight.grad.abs().max().item(), 0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        output = block(x, c)[0]
        output.square().mean().backward()  # NO regularizer: gradient must traverse selection.
        self.assertGreater(block.selector[-1].weight.grad.abs().max().item(), 0)
        for name, p in block.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_checkpoint_roundtrip_and_autocast(self):
        cascade = self.naf.CascadeDiscreteNAF2D(8).eval()
        for block in cascade.stages.values():
            nn.init.normal_(block.output_projection.weight, std=.1)
        features = [torch.randn(2, 8, h, h) for h in (32, 16, 8, 4)]
        before = cascade(features)[0]
        buffer = io.BytesIO(); torch.save(cascade.state_dict(), buffer); buffer.seek(0)
        restored = self.naf.CascadeDiscreteNAF2D(8).eval()
        restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
        for a, b in zip(before, restored(features)[0]):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = restored([x.bfloat16() for x in features])
        self.assertTrue(torch.isfinite(output[1]))
        self.assertTrue(all(x.dtype == torch.bfloat16 and torch.isfinite(x).all() for x in output[0]))

    @unittest.skipUnless(os.environ.get('EXP09_FULL_SMOKE') == '1', 'optional real Hiera check not requested')
    def test_full_pretrained_hiera_batch12_two_synthetic_steps(self):
        checkpoint = os.environ.get('HIERA_CHECKPOINT',
            '/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors')
        self.assertTrue(Path(checkpoint).is_file(), checkpoint)
        device = torch.device(os.environ.get('EXP09_DEVICE', 'cuda'))
        batch = int(os.environ.get('EXP09_SMOKE_BATCH', '12'))
        self.assertGreater(batch, 0)
        previous = {name: sys.modules.pop(name) for name in list(sys.modules)
                    if name == 'networks' or name.startswith('networks.')}
        sys.path.insert(0, str(self.source))
        try:
            from networks.mslau_net import MSLAU_net
            from loss import StructureLoss_binary
            torch.manual_seed(1234)
            model = MSLAU_net(img_size=352, decoder_mode='p5_sam2unet',
                              encoder_name='sam2_hiera_large', hiera_checkpoint=checkpoint).to(device)
            self.assertTrue(model.encoder.freeze_backbone)
            names = [name for name, p in model.encoder.named_parameters() if p.requires_grad]
            self.assertTrue(names)
            self.assertTrue(all('.prompt_learn.' in name for name in names))
            groups = [[p for p in model.encoder.parameters() if p.requires_grad],
                      [p for name, p in model.named_parameters() if not name.startswith('encoder.')]]
            optimizer = torch.optim.AdamW([dict(params=p, lr=1e-3) for p in groups], weight_decay=5e-4)
            owned = {id(p) for g in optimizer.param_groups for p in g['params']}
            enhancer = model.progressive_decoder.frequency_enhancer
            self.assertEqual(set(enhancer.stages), {'p1', 'p2', 'p3'})
            self.assertTrue(all(id(p) in owned for p in enhancer.parameters()))
            enhancer.set_selection_context(1234, 0, ['synthetic_%d' % i for i in range(batch)])
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            image = torch.randn(batch, 3, 352, 352, device=device)
            target = torch.randint(0, 2, (batch, 1, 352, 352), device=device).float()
            criterion = StructureLoss_binary()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                output = model(image, return_aux=True)
                loss = sum(criterion(output[k], target) for k in ('logits', 'aux_d2', 'aux_d3'))
                objective = loss + .001 * output['naf_regularization'] if step == 0 else loss
                objective.backward()
                self.assertTrue(torch.isfinite(objective))
                for name, p in enhancer.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                if step == 0:
                    optimizer.step()
                else:
                    for block in enhancer.stages.values():
                        self.assertGreater(block.selector[-1].weight.grad.abs().max().item(), 0)
            summary = dict(batch=batch, device=str(device), added_parameters=56118,
                           regularizer=output['naf_regularization'].item(),
                           task_loss=loss.item(), selector_task_gradients='finite_nonzero',
                           retained_full_fractions=output['naf_stats'][..., 3].detach().cpu().tolist())
            if device.type == 'cuda':
                summary['peak_allocated_MiB'] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            print('EXP09_FULL_SMOKE:', json.dumps(summary), flush=True)
        finally:
            sys.path.remove(str(self.source))
            for name in list(sys.modules):
                if name == 'networks' or name.startswith('networks.'):
                    del sys.modules[name]
            sys.modules.update(previous)


@unittest.skipIf(torch is None or A is None, 'real PyTorch/Albumentations required')
class DiscretePairedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix='exp09_inputs_')
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / 'source'
        builder = load('exp09_inputs_builder', ROOT / 'tools/build_experiment.py')
        builder.build(EXPERIMENT, cls.source)
        cls.previous_loader = sys.modules.pop('loader', None)
        sys.path.insert(0, str(cls.source))
        cls.paired = load('exp09_paired_inputs', cls.source / 'paired_inputs.py')
        cls.naf = load('exp09_paired_naf', cls.source / 'networks/naf_discrete.py')
        cls.factories = definitions(cls.source / 'kvasir_train.py',
            {'get_train_transform', 'get_valid_transform'}, dict(A=A, ToTensorV2=ToTensorV2))
        cls.dataset = cls.root / 'data'
        (cls.dataset / 'images').mkdir(parents=True)
        (cls.dataset / 'masks').mkdir()
        cls.ids = ['%d.png' % i for i in range(9)]
        for i, name in enumerate(cls.ids):
            image = np.random.default_rng(i).integers(0, 256, (37, 43, 3), dtype=np.uint8)
            mask = np.zeros((37, 43), dtype=np.uint8); mask[3:29, 5 + i:31] = 255
            Image.fromarray(image).save(cls.dataset / 'images' / name)
            Image.fromarray(mask).save(cls.dataset / 'masks' / name)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.source))
        sys.modules.pop('loader', None)
        if cls.previous_loader is not None:
            sys.modules['loader'] = cls.previous_loader
        cls.temporary.cleanup()

    def loaders(self):
        return {phase: self.paired.make_loader(self.paired.PairedDataset(str(self.dataset), self.ids,
            self.factories['get_train_transform' if phase == 'train' else 'get_valid_transform'](32),
            1234, phase), 4, 0) for phase in ('train', 'valid')}

    def test_paired_inputs_match_baseline_protocol(self):
        baseline = ROOT / 'experiments/exp00_baseline_paired_aug_v1/added/paired_inputs.py'
        original = ast.parse(baseline.read_text())
        actual = ast.parse((self.source / 'paired_inputs.py').read_text())
        # The only helper difference adds a common-baseline state hash; all
        # augmentation/sampling/digest definitions are exactly the same AST.
        a = [n for n in original.body if not isinstance(n, ast.FunctionDef) or n.name != 'record_initial_state']
        b = [n for n in actual.body if not isinstance(n, ast.FunctionDef) or n.name != 'record_initial_state']
        self.assertEqual(ast.dump(ast.Module(body=a, type_ignores=[])), ast.dump(ast.Module(body=b, type_ignores=[])))
        for epoch in (0, 1):
            first, second = self.loaders()['train'], self.loaders()['train']
            self.paired.set_loader_epoch(first, epoch); self.paired.set_loader_epoch(second, epoch)
            a, b = self.paired.InputDigest(), self.paired.InputDigest()
            for images, masks, ids in first:
                a.update(images, masks, ids)
                self.naf.sample_gumbels(1234, epoch, ids, 1, 'local', 60, 'cpu', torch.float32)
            torch.rand(97)
            for images, masks, ids in second:
                b.update(images, masks, ids)
            self.assertEqual(a.record(1234, epoch, 'train'), b.record(1234, epoch, 'train'))

    def test_actual_loop_sets_selection_context_and_saves_three_weights(self):
        namespace = definitions(self.source / 'kvasir_train.py',
            {'train_model', 'snapshot_state_dict', 'atomic_save_state_dict', 'log_fusion_scale_stats'},
            dict(torch=torch, nn=nn, Variable=torch.autograd.Variable, copy=copy, time=time, os=os,
                 logging=logging, BRANCHES=self.naf.BRANCHES,
                 FILTER_STATS_NAMES=self.naf.FILTER_STATS_NAMES, FUSION_STATS_NAMES=self.naf.FUSION_STATS_NAMES,
                 **{k: getattr(self.paired, k) for k in ('set_loader_epoch', 'InputDigest', 'append_record')}))
        naf = self.naf
        class TinyModel(nn.Module):
            decoder_mode = 'p5_sam2unet'
            conv_mla = None
            naf_reg_weight = .001
            def __init__(self):
                super().__init__()
                self.projection = nn.Conv2d(3, 4, 1)
                self.progressive_decoder = nn.Module()
                self.progressive_decoder.frequency_enhancer = naf.CascadeDiscreteNAF2D(4)
                self.head = nn.Conv2d(4, 1, 1)
                self.calls = []
            def forward(self, images, **kwargs):
                projected = self.projection(images)
                features = [F.interpolate(projected, size=(h, h)) for h in (32, 16, 8, 4)]
                enhanced, reg, stats, fusion = self.progressive_decoder.frequency_enhancer(features)
                self.calls.append((self.training, images.shape[0], reg.detach().item()))
                logits = self.head(enhanced[0])
                return dict(logits=logits, aux_d2=logits, aux_d3=logits,
                            naf_regularization=reg, naf_stats=stats, naf_fusion_stats=fusion)
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / 'logs'; logs.mkdir()
            model = TinyModel()
            initial = self.paired.record_initial_state(model, 1234, logs)
            base_state = {name: value for name, value in model.state_dict().items()
                          if not name.startswith('progressive_decoder.frequency_enhancer.')}
            self.assertEqual(initial['baseline_state_sha256'], self.paired.state_digest(base_state))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
            with patch('torch.cuda.is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
                namespace['train_model'](model, nn.BCEWithLogitsLoss(), optimizer, scheduler,
                    self.loaders(), lambda a, b: (a.sigmoid() - b).abs().mean(), 2, tmp,
                    aux_d2_weight=1, aux_d3_weight=1,
                    input_audit_path=str(logs / 'input_fingerprints.jsonl'))
            rows = [json.loads(l) for l in (logs / 'input_fingerprints.jsonl').read_text().splitlines()]
            self.assertEqual([(r['epoch'], r['phase']) for r in rows],
                             [(0, 'train'), (0, 'valid'), (1, 'train'), (1, 'valid')])
            self.assertEqual(len(list(Path(tmp).glob('*.pth'))), 3)
            metric_rows = [json.loads(l) for l in (logs / 'epoch_metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(metric_rows), 4)
            self.assertTrue(all(math.isfinite(r['loss']) for r in metric_rows))
            offset = 0
            for row in metric_rows:
                count = next(r['batches'] for r in rows if r['epoch'] == row['epoch'] and r['phase'] == row['phase'])
                calls = model.calls[offset:offset + count]; offset += count
                mean_reg = sum(n * r for _, n, r in calls) / sum(n for _, n, _ in calls)
                expected = 3 * row['main_loss'] + (.001 * mean_reg if row['phase'] == 'train' else 0)
                self.assertAlmostEqual(row['loss'], expected, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
