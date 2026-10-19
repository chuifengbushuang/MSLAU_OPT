"""Exp01 numerical and rendered-source integration checks; no dataset training.

CPU: python -m unittest discover -s tests -p 'test_exp01_naf.py' -v
Optional server full model: NAF_FULL_SMOKE=1 NAF_DEVICE=cuda <same command>
"""
import ast
import copy
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/exp00_baseline'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definitions(path, names, namespace):
    """Execute the actual rendered definitions, without optional encoder imports."""
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError('Missing source definition')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@unittest.skipIf(torch is None, 'torch is required for numerical NAF checks')
class NAFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix='exp01_test_')
        cls.source = Path(cls.temporary.name) / 'source'
        builder = load('exp01_builder', ROOT / 'tools/build_experiment.py')
        cls.build_manifest = builder.build('exp01_naf_select_reg', cls.source)
        cls.naf = load('exp01_naf', cls.source / 'networks/naf_select_reg.py')
        cls.class_names = ['BasicConv2d', 'RFBModified', 'SAM2UNetUp', 'SAM2UNetDecoder']
        cls.original = definitions(BASE / 'networks/mslau_net.py', cls.class_names,
                                   dict(torch=torch, nn=nn, F=F))
        cls.rendered = definitions(cls.source / 'networks/mslau_net.py', cls.class_names,
                                   dict(torch=torch, nn=nn, F=F, BudgetNAF2D=cls.naf.BudgetNAF2D))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_budget_bounds_dc_and_nonuniform_selection(self):
        for height, width in [(11, 11), (22, 22), (13, 18), (12, 17)]:
            values = torch.rand(3, 1, height, width // 2 + 1, dtype=torch.double)
            weights = self.naf.frequency_geometry(height, width, values.device, values.dtype)[-1]
            gate = self.naf.soft_budget_mask(values, weights)
            self.assertGreaterEqual(gate.min().item(), 0)
            self.assertLessEqual(gate.max().item(), 1)
            torch.testing.assert_close(gate[..., 0, 0], torch.ones(3, 1, dtype=torch.double))
            torch.testing.assert_close((gate * weights).sum((1, 2, 3)) / weights.sum(),
                                       torch.full((3,), 0.9, dtype=torch.double))
            self.assertGreater(gate.std().item(), 0.01)

    def test_budget_analytic_gradient(self):
        torch.manual_seed(4)
        values = torch.rand(1, 1, 3, 4, dtype=torch.double, requires_grad=True)
        weights = torch.ones_like(values)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: self.naf.soft_budget_mask(x, weights), (values,), atol=1e-4))
        # Include excluded DC, with the DC score sorted last (zero suffix mass).
        weights[..., 0, 0] = 0
        values = values.detach().clone()
        values[..., 0, 0] = 0
        values.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: self.naf.soft_budget_mask(x, weights), (values,), atol=1e-4))

    def test_piecewise_penalty_has_frequency_and_stage_effect(self):
        _, _, bands, weights = self.naf.frequency_geometry(22, 22, 'cpu', torch.double)
        values = torch.full_like(weights, 0.5, requires_grad=True)
        gate = self.naf.soft_budget_mask(values, weights)
        coefficient3 = 3.0 ** (-(bands + 1).to(torch.double))
        coefficient4 = 4.0 ** (-(bands + 1).to(torch.double))
        penalty = (gate.square() * coefficient3 * weights).sum() / weights.sum()
        penalty4 = (gate.square() * coefficient4 * weights).sum() / weights.sum()
        gradient, = torch.autograd.grad(penalty, values)
        self.assertGreater(penalty.item(), penalty4.item())
        self.assertGreater(gradient[(bands == 0) & (weights > 0)].mean().item(), 0)
        self.assertLess(gradient[bands == 2].mean().item(), 0)

    def test_real_spectrum_symmetry_constant_preservation_and_energy(self):
        module = self.naf.BudgetNAF2D(3)
        # Test a nonuniform learned selector, rather than only its flat init.
        nn.init.normal_(module.selector[-1].weight, std=0.3)
        for height, width in [(11, 11), (22, 22), (13, 18), (12, 17)]:
            features = torch.randn(2, 4, height, width)
            spectrum = torch.fft.rfft2(features, norm='ortho')
            gate, _, _ = module.selection(spectrum, (height, width))
            filtered = torch.fft.irfft2(spectrum * gate, s=(height, width), norm='ortho')
            torch.testing.assert_close(torch.fft.rfft2(filtered, norm='ortho'), spectrum * gate,
                                       atol=2e-6, rtol=2e-5)
            output, _, _ = module(features)
            torch.testing.assert_close(output.mean((-2, -1)), features.mean((-2, -1)), atol=1e-6, rtol=1e-5)
            self.assertLessEqual(output.square().sum().item(), features.square().sum().item() + 1e-3)
            constant = torch.full_like(features, 3.0)
            torch.testing.assert_close(module(constant)[0], constant)

    def test_finite_gradients_and_autocast(self):
        for dtype in (torch.float32, torch.bfloat16):
            module = self.naf.BudgetNAF2D(4)
            features = torch.randn(2, 4, 11, 11, dtype=dtype, requires_grad=True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                output, penalty, _ = module(features)
                loss = output.float().square().mean() + 1e-3 * penalty
            loss.backward()
            self.assertEqual(output.dtype, dtype)
            self.assertTrue(torch.isfinite(features.grad).all())
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(module.selector[-1].weight.grad.abs().max().item(), 0)

    def test_original_decoder_initialization_and_rng_unchanged(self):
        options = dict(encoder_channels=(4, 8, 16, 32), channels=8)
        torch.manual_seed(1234)
        original = self.original['SAM2UNetDecoder'](**options)
        original_rng = torch.get_rng_state()
        torch.manual_seed(1234)
        experiment = self.rendered['SAM2UNetDecoder'](**options)
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(experiment.state_dict()[name], value, rtol=0, atol=0)
        new_names = set(experiment.state_dict()) - set(original.state_dict())
        self.assertTrue(all(name.startswith(('naf_p3.', 'naf_p4.')) for name in new_names))
        self.assertEqual(sum(p.numel() for p in experiment.parameters()) -
                         sum(p.numel() for p in original.parameters()), 164)

    def test_rendered_decoder_aux_optimizer_and_strict_reload(self):
        decoder_class = self.rendered['SAM2UNetDecoder']
        options = dict(encoder_channels=(4, 8, 16, 32), channels=8)
        model = decoder_class(**options)
        inputs = [torch.randn(2, c, s, s) for c, s in zip((4, 8, 16, 32), (88, 44, 22, 11))]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        parameter_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in parameter_ids for p in model.naf_p3.parameters()))
        output = model(inputs, 352, return_aux=True)
        for name in ['logits', 'aux_d2', 'aux_d3']:
            self.assertEqual(output[name].shape, (2, 1, 352, 352))
        self.assertEqual(output['naf_stats'].shape, (2, 7))
        (sum(output[key].square().mean() for key in ['logits', 'aux_d2', 'aux_d3']) +
         1e-3 * output['naf_regularization']).backward()
        self.assertIsNotNone(model.naf_p4.selector[-1].weight.grad)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = decoder_class(**options)
        restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(inputs, 352), restored(inputs, 352), rtol=0, atol=0)

    def test_rendered_training_loss_and_three_checkpoint_contract(self):
        namespace = dict(torch=torch, nn=nn, F=F, copy=copy, os=os, time=time, logging=logging,
                         Variable=torch.autograd.Variable, STATS_NAMES=self.naf.STATS_NAMES,
                         log_fusion_scale_stats=lambda *args: None)
        definitions(self.source / 'kvasir_train.py',
                    ['snapshot_state_dict', 'atomic_save_state_dict', 'train_model', 'make_boundary_target'], namespace)
        naf_type = self.naf.BudgetNAF2D

        class TinyModel(nn.Module):
            decoder_mode = 'p5_sam2unet'
            naf_reg_weight = 1.0

            def __init__(self):
                super().__init__()
                self.naf = naf_type(3)

            def forward(self, x, **kwargs):
                output, penalty, stats = self.naf(x)
                return dict(logits=output, naf_regularization=penalty,
                            naf_stats=torch.stack([stats, stats]))

        model = TinyModel()
        features = torch.randn(2, 1, 11, 11)
        labels = torch.zeros_like(features)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
        batches = [(features, labels, ['a', 'b'])]
        with tempfile.TemporaryDirectory() as tmp, patch.object(logging, 'info') as log, \
             patch.object(torch.cuda, 'is_available', return_value=False):
            namespace['train_model'](model, nn.BCEWithLogitsLoss(), optimizer, scheduler,
                                     dict(train=batches, valid=batches),
                                     lambda x, y: x.new_tensor(0.5), 1, tmp, boundary_weight=0)
            self.assertEqual(len(list(Path(tmp).glob('*.pth'))), 3)
            lines = [call.args[0] for call in log.call_args_list if call.args]
            train = next(line for line in lines if line.startswith('train Loss:'))
            valid = next(line for line in lines if line.startswith('valid Loss:'))
            self.assertGreater(float(train.split()[2]), float(valid.split()[2]))
            self.assertEqual(train.split()[4], valid.split()[4])

    @unittest.skipUnless(os.environ.get('NAF_FULL_SMOKE') == '1', 'optional real Hiera smoke not requested')
    def test_full_hiera_model(self):
        # This path is explicitly opt-in and uses only synthetic tensors.
        checkpoint = os.environ.get('HIERA_CHECKPOINT',
            '/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors')
        self.assertTrue(Path(checkpoint).is_file(), checkpoint)
        device = torch.device(os.environ.get('NAF_DEVICE', 'cuda'))
        sys.path.insert(0, str(self.source))
        try:
            from networks.mslau_net import MSLAU_net
            model = MSLAU_net(img_size=352, decoder_mode='p5_sam2unet',
                              encoder_name='sam2_hiera_large', hiera_checkpoint=checkpoint).to(device)
            self.assertTrue(model.encoder.freeze_backbone)
            encoder_trainable = [name for name, p in model.encoder.named_parameters() if p.requires_grad]
            self.assertTrue(encoder_trainable)
            self.assertTrue(all('.prompt_learn.' in name for name in encoder_trainable))
            groups = [[p for p in model.encoder.parameters() if p.requires_grad],
                      [p for name, p in model.named_parameters() if not name.startswith('encoder.')]]
            optimizer = torch.optim.AdamW([dict(params=params, lr=1e-3) for params in groups])
            owned = {id(p) for group in optimizer.param_groups for p in group['params']}
            self.assertTrue(all(id(p) in owned for p in model.progressive_decoder.naf_p3.parameters()))
            output = model(torch.randn(2, 3, 352, 352, device=device), return_aux=True)
            loss = sum(output[key].square().mean() for key in ['logits', 'aux_d2', 'aux_d3'])
            (loss + 1e-3 * output['naf_regularization']).backward()
            for module in [model.progressive_decoder.naf_p3, model.progressive_decoder.naf_p4]:
                for p in module.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
        finally:
            sys.path.remove(str(self.source))
            for name in list(sys.modules):
                if name == 'networks' or name.startswith('networks.'):
                    del sys.modules[name]


if __name__ == '__main__':
    unittest.main()
