"""Experiment 7 numerical and real rendered-source checks; no dataset training.

Optional 3090 check: EXP07_FULL_SMOKE=1 EXP07_DEVICE=cuda EXP07_SMOKE_BATCH=12
python -m unittest discover -s tests -p 'test_exp07_naf.py' -v
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
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/exp00_baseline'
EXPERIMENT = 'exp07_naf_cascade'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError('Missing actual source definition')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@unittest.skipIf(torch is None, 'torch is required for numerical NAF checks')
class CascadeNAFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix='exp07_test_')
        cls.source = Path(cls.temporary.name) / 'source'
        builder = load('exp07_builder', ROOT / 'tools/build_experiment.py')
        cls.manifest = builder.build(EXPERIMENT, cls.source)
        cls.naf = load('exp07_naf', cls.source / 'networks/naf_cascade.py')
        names = ['BasicConv2d', 'RFBModified', 'SAM2UNetUp', 'SAM2UNetDecoder']
        cls.original = definitions(BASE / 'networks/mslau_net.py', names, dict(torch=torch, nn=nn, F=F))
        cls.rendered = definitions(cls.source / 'networks/mslau_net.py', names,
                                   dict(torch=torch, nn=nn, F=F, CascadeNAF2D=cls.naf.CascadeNAF2D))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_config_matches_baseline_and_standalone_preview(self):
        config = json.loads((self.source / 'config.json').read_text())
        baseline = json.loads((BASE / 'config.json').read_text())
        options = config['train_options'].copy()
        self.assertEqual(options.pop('naf_reg_weight'), 0.001)
        self.assertEqual(options, baseline['train_options'])
        self.assertEqual(config['eval_options'], baseline['eval_options'])
        self.assertEqual(set(self.manifest['added_files']), {'networks/naf_cascade.py'})
        launcher = load('exp07_launcher', self.source / 'run.py')
        train, evaluate = launcher.commands(1234, 'preview')
        self.assertIn(EXPERIMENT, config['id'])
        self.assertIn('--naf_reg_weight', train)
        self.assertNotIn('--naf_stages', train + evaluate)
        self.assertEqual(evaluate[evaluate.index('--model') + 1].split('/')[-1], 'kvasir_best_model.pth')
        # Importable, ordinary rendered source; no dependency on older NAF code.
        self.assertNotIn('naf_select_reg', (self.source / 'networks/mslau_net.py').read_text())

    def test_two_separate_budgets_and_analytic_gradient(self):
        for h, w in ((11, 11), (22, 22), (13, 18), (12, 17)):
            weights = self.naf.frequency_geometry(h, w, 'cpu', torch.double)[-1]
            # Distinct branch logits must still each have a 0.9 budget.
            values = torch.rand(6, 1, h, w // 2 + 1, dtype=torch.double)
            gate = self.naf.soft_budget_mask(values, weights)
            torch.testing.assert_close((gate * weights).sum((1, 2, 3)) / weights.sum(),
                                       torch.full((6,), 0.9, dtype=torch.double))
            self.assertTrue(((gate >= 0) & (gate <= 1)).all())
            torch.testing.assert_close(gate[..., 0, 0], torch.ones(6, 1, dtype=torch.double))
        values = torch.rand(2, 1, 3, 4, dtype=torch.double, requires_grad=True)
        weights = torch.ones(1, 1, 3, 4, dtype=torch.double)
        weights[..., 0, 0] = 0
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: self.naf.soft_budget_mask(x, weights), (values,), atol=1e-4))

    def test_joint_selector_uses_context_for_both_masks(self):
        torch.manual_seed(7)
        block = self.naf.DualInputNAF2D(3, 4)
        nn.init.normal_(block.selector[-1].weight, std=0.3)
        local = torch.randn(2, 4, 22, 22)
        context_a = torch.randn_like(local)
        context_b = torch.randn_like(local)
        local_fft = torch.fft.rfft2(local, norm='ortho')
        a = block.selection(local_fft, torch.fft.rfft2(context_a, norm='ortho'), (22, 22))[0]
        b = block.selection(local_fft, torch.fft.rfft2(context_b, norm='ortho'), (22, 22))[0]
        for branch in range(2):
            self.assertGreater((a[:, branch] - b[:, branch]).abs().max().item(), 1e-4)

    def test_filters_preserve_dc_real_spectrum_and_do_not_amplify_energy(self):
        torch.manual_seed(9)
        block = self.naf.DualInputNAF2D(2, 4)
        nn.init.normal_(block.selector[-1].weight, std=0.3)
        for h, w in ((11, 11), (22, 22), (13, 18), (12, 17)):
            inputs = [torch.randn(2, 4, h, w) for _ in range(2)]
            spectra = [torch.fft.rfft2(x, norm='ortho') for x in inputs]
            gates, _, weights = block.selection(*spectra, (h, w))
            filtered = block.filter_pair(*inputs)[:2]
            for branch in range(2):
                target = spectra[branch] * gates[:, branch:branch + 1]
                reconstructed = torch.fft.irfft2(target, s=(h, w), norm='ortho')
                torch.testing.assert_close(torch.fft.rfft2(reconstructed, norm='ortho'), target,
                                           atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(filtered[branch].mean((-2, -1)), inputs[branch].mean((-2, -1)),
                                           atol=1e-6, rtol=1e-5)
                self.assertLessEqual(filtered[branch].square().sum().item(),
                                     inputs[branch].square().sum().item() + 1e-3)
                torch.testing.assert_close((gates[:, branch:branch + 1] * weights).sum((1, 2, 3)) /
                                           weights.sum(), torch.full((2,), 0.9))
            constant = torch.full_like(inputs[0], 3)
            for value in block.filter_pair(constant, constant)[:2]:
                torch.testing.assert_close(value, constant)

    def test_fusion_summary_detects_non_dc_changes_that_gap_cannot(self):
        block = self.naf.DualInputNAF2D(2, 4)
        x = torch.randn(2, 4, 22, 22) + 2
        filtered = block.filter_pair(x, x)[0]
        before, after = self.naf.spatial_summary(x), self.naf.spatial_summary(filtered)
        torch.testing.assert_close(before[:, :4], after[:, :4], atol=1e-6, rtol=1e-5)
        self.assertGreater((before[:, 4:] - after[:, 4:]).abs().max().item(), 1e-4)

    def test_six_term_penalty_uses_input_labels_and_fixed_denominator(self):
        block = self.naf.DualInputNAF2D(2, 4)
        nn.init.normal_(block.selector[-1].weight, std=0.2)
        x, c = torch.randn(2, 4, 22, 22), torch.randn(2, 4, 22, 22)
        spectra = [torch.fft.rfft2(t, norm='ortho') for t in (x, c)]
        gates, bands, weights = block.selection(*spectra, (22, 22))
        penalties = block.filter_pair(x, c)[2]
        for branch, n in enumerate((2, 3)):
            expected = (gates[:, branch:branch + 1].square() * weights *
                        n ** (-(bands + 1).float())).sum() / (2 * weights.sum())
            torch.testing.assert_close(penalties[branch], expected)
        # Capture the real branch outputs and inputs; this detects accidentally
        # using P3/P2 instead of cascaded F3/F2, or detaching the context.
        cascade = self.naf.CascadeNAF2D(4)
        features = [torch.randn(2, 4, h, h) for h in (88, 44, 22, 11)]
        captured, order, handles = {}, [], []
        for level in (1, 2, 3):
            module = cascade.stages['p%d' % level]
            nn.init.normal_(module.output_projection.weight, std=0.1)

            def record(module, inputs, outputs, level=level):
                captured[level] = (inputs, outputs)
                order.append(level)

            handles.append(module.register_forward_hook(record))
        try:
            enhanced, penalty, stats, fusion = cascade(features)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(order, [3, 2, 1])
        self.assertEqual(set(cascade.stages), {'p1', 'p2', 'p3'})
        self.assertIs(captured[3][0][1], features[3])
        self.assertIs(captured[2][0][1], captured[3][1][0])
        self.assertIs(captured[1][0][1], captured[2][1][0])
        self.assertIs(enhanced[3], features[3])
        torch.testing.assert_close(penalty, sum(captured[i][1][1].sum() for i in (1, 2, 3)) / 6)
        self.assertEqual(stats.shape, (3, 2, len(self.naf.FILTER_STATS_NAMES)))
        self.assertEqual(fusion.shape, (3, len(self.naf.FUSION_STATS_NAMES)))
        # A shallow output really backpropagates through the deepest new block.
        enhanced[0].square().mean().backward()
        gradient = cascade.stages['p3'].output_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().max().item(), 0)

    def test_initial_decoder_outputs_original_parameters_and_rng_match(self):
        options = dict(encoder_channels=(4, 8, 16, 32), channels=8)
        torch.manual_seed(1234)
        original = self.original['SAM2UNetDecoder'](**options)
        original_rng = torch.get_rng_state()
        torch.manual_seed(1234)
        model = self.rendered['SAM2UNetDecoder'](**options)
        torch.testing.assert_close(torch.get_rng_state(), original_rng, atol=0, rtol=0)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        new = set(model.state_dict()) - set(original.state_dict())
        self.assertTrue(new)
        self.assertTrue(all(name.startswith('frequency_enhancer.stages.') for name in new))
        inputs = [torch.randn(2, c, h, h) for c, h in zip((4, 8, 16, 32), (88, 44, 22, 11))]
        for training in (False, True):
            original.train(training)
            model.train(training)
            with torch.no_grad():
                expected, actual = original(inputs, 352, True), model(inputs, 352, True)
            for key in ('logits', 'aux_d2', 'aux_d3'):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_zero_projection_wakes_up_and_strict_checkpoint_roundtrip(self):
        torch.manual_seed(17)
        block = self.naf.DualInputNAF2D(2, 8)
        local, context = torch.randn(2, 8, 22, 22), torch.randn(2, 8, 11, 11)
        optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
        output = block(local, context)[0]
        torch.testing.assert_close(output, local, atol=0, rtol=0)
        for step in range(3):
            optimizer.zero_grad()
            output, penalty, _, _ = block(local, context)
            (output.square().mean() + 1e-3 * penalty.sum() / 6).backward()
            for name, parameter in block.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            if step == 0:
                self.assertGreater(block.output_projection.weight.grad.abs().max().item(), 0)
            if step == 2:
                for name in ('local_projection', 'context_projection', 'output_projection'):
                    self.assertGreater(getattr(block, name).weight.grad.abs().max().item(), 0, name)
                self.assertGreater(block.fusion_gate[-1].weight.grad.abs().max().item(), 0)
                self.assertGreater(block.selector[-1].weight.grad.abs().max().item(), 0)
            optimizer.step()
        a = block(local, context)[0]
        b = block(local, -context)[0]
        self.assertGreater((a - b).abs().max().item(), 1e-5)
        buffer = io.BytesIO()
        torch.save(block.state_dict(), buffer)
        buffer.seek(0)
        restored = self.naf.DualInputNAF2D(2, 8)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(restored(local, context)[0], a, atol=0, rtol=0)

    def test_autocast_and_real_64_channel_pyramid_backward(self):
        cascade = self.naf.CascadeNAF2D(64)
        features = [torch.randn(2, 64, h, h, dtype=torch.bfloat16, requires_grad=True)
                    for h in (88, 44, 22, 11)]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            enhanced, penalty, stats, fusion = cascade(features)
            loss = sum(t.float().square().mean() for t in enhanced) + 1e-3 * penalty
        loss.backward()
        self.assertTrue(all(t.dtype == torch.bfloat16 for t in enhanced))
        self.assertTrue(torch.isfinite(stats).all() and torch.isfinite(fusion).all())
        for t in features:
            self.assertTrue(torch.isfinite(t.grad).all())
        for name, parameter in cascade.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_rendered_decoder_optimizer_and_strict_reload(self):
        options = dict(encoder_channels=(4, 8, 16, 32), channels=8)
        model = self.rendered['SAM2UNetDecoder'](**options)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        owned = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in owned for p in model.frequency_enhancer.parameters()))
        inputs = [torch.randn(2, c, h, h) for c, h in zip((4, 8, 16, 32), (88, 44, 22, 11))]
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            output = model(inputs, 352, True)
            loss = sum(output[key].square().mean() for key in ('logits', 'aux_d2', 'aux_d3'))
            (loss + 1e-3 * output['naf_regularization']).backward()
            optimizer.step()
        restored = self.rendered['SAM2UNetDecoder'](**options)
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        with self.assertRaises(RuntimeError):
            # A failing strict load can already copy matching tensors. Test
            # rejection on a separate instance, not the roundtrip target.
            self.rendered['SAM2UNetDecoder'](**options).load_state_dict(
                self.original['SAM2UNetDecoder'](**options).state_dict(), strict=True)
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(inputs, 352), restored(inputs, 352), atol=0, rtol=0)

    def test_actual_training_regularizer_validation_and_three_checkpoints(self):
        namespace = dict(torch=torch, nn=nn, F=F, copy=copy, os=os, time=time, logging=logging,
                         Variable=torch.autograd.Variable, BRANCHES=self.naf.BRANCHES,
                         FILTER_STATS_NAMES=self.naf.FILTER_STATS_NAMES,
                         FUSION_STATS_NAMES=self.naf.FUSION_STATS_NAMES,
                         log_fusion_scale_stats=lambda *args: None)
        definitions(self.source / 'kvasir_train.py',
                    ['snapshot_state_dict', 'atomic_save_state_dict', 'train_model', 'make_boundary_target'], namespace)
        naf_type = self.naf.CascadeNAF2D

        class TinyModel(nn.Module):
            decoder_mode = 'p5_sam2unet'
            naf_reg_weight = 1.0

            def __init__(self):
                super().__init__()
                self.progressive_decoder = nn.Module()
                self.progressive_decoder.frequency_enhancer = naf_type(1)

            def forward(self, x, **kwargs):
                features, penalty, stats, fusion = self.progressive_decoder.frequency_enhancer([x] * 4)
                return dict(logits=features[0], naf_regularization=penalty,
                            naf_stats=stats, naf_fusion_stats=fusion)

        inputs = torch.randn(2, 1, 11, 11)
        batches = [(inputs, torch.zeros_like(inputs), ['a', 'b'])]
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(logging, 'info') as log, \
             patch.object(torch.cuda, 'is_available', return_value=False):
            namespace['train_model'](model, nn.BCEWithLogitsLoss(), optimizer, scheduler,
                                     dict(train=batches, valid=batches),
                                     lambda x, y: x.new_tensor(0.5), 1, tmp, boundary_weight=0)
            files = {p.name for p in Path(tmp).glob('*.pth')}
            self.assertEqual(len(files), 3)
            self.assertIn('kvasir_best_model.pth', files)
            self.assertTrue(any(name.startswith('best_iou_') for name in files))
            self.assertTrue(any(name.startswith('best_loss_') for name in files))
            lines = [call.args[0] for call in log.call_args_list if call.args]
            train = next(line for line in lines if line.startswith('train Loss:'))
            valid = next(line for line in lines if line.startswith('valid Loss:'))
            self.assertGreater(float(train.split()[2]), float(valid.split()[2]))
            self.assertEqual(train.split()[4], valid.split()[4])
            self.assertEqual(sum(line == '%s NAF%d %s: %s' for line in lines), 12)
            self.assertEqual(sum(line == '%s NAF%d fusion: %s' for line in lines), 6)

    @unittest.skipUnless(os.environ.get('EXP07_FULL_SMOKE') == '1', 'optional real Hiera smoke not requested')
    def test_full_pretrained_hiera_batch12_forward_backward(self):
        checkpoint = os.environ.get('HIERA_CHECKPOINT',
            '/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors')
        self.assertTrue(Path(checkpoint).is_file(), checkpoint)
        device = torch.device(os.environ.get('EXP07_DEVICE', 'cuda'))
        batch = int(os.environ.get('EXP07_SMOKE_BATCH', '12'))
        self.assertGreater(batch, 0)
        # Clear modules from an earlier optional v1 test, then load actual source.
        previous = {name: sys.modules.pop(name) for name in list(sys.modules)
                    if name == 'networks' or name.startswith('networks.')}
        sys.path.insert(0, str(self.source))
        try:
            from networks.mslau_net import MSLAU_net
            from loss import StructureLoss_binary
            model = MSLAU_net(img_size=352, decoder_mode='p5_sam2unet',
                              encoder_name='sam2_hiera_large', hiera_checkpoint=checkpoint).to(device)
            self.assertTrue(model.encoder.freeze_backbone)
            trainable = [name for name, p in model.encoder.named_parameters() if p.requires_grad]
            self.assertTrue(trainable)
            self.assertTrue(all('.prompt_learn.' in name for name in trainable))
            groups = [[p for p in model.encoder.parameters() if p.requires_grad],
                      [p for name, p in model.named_parameters() if not name.startswith('encoder.')]]
            optimizer = torch.optim.AdamW([dict(params=p, lr=1e-3) for p in groups], weight_decay=5e-4)
            owned = {id(p) for group in optimizer.param_groups for p in group['params']}
            enhancer = model.progressive_decoder.frequency_enhancer
            self.assertEqual(set(enhancer.stages), {'p1', 'p2', 'p3'})
            self.assertTrue(all(id(p) in owned for p in enhancer.parameters()))
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            output = model(torch.randn(batch, 3, 352, 352, device=device), return_aux=True)
            target = torch.randint(0, 2, (batch, 1, 352, 352), device=device).float()
            criterion = StructureLoss_binary()
            loss = sum(criterion(output[key], target) for key in ('logits', 'aux_d2', 'aux_d3'))
            (loss + 1e-3 * output['naf_regularization']).backward()
            self.assertTrue(torch.isfinite(loss))
            for name, parameter in enhancer.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for block in enhancer.stages.values():
                self.assertGreater(block.output_projection.weight.grad.abs().max().item(), 0)
            summary = dict(batch=batch, device=str(device),
                           added_parameters=sum(p.numel() for p in enhancer.parameters()),
                           regularizer=output['naf_regularization'].item())
            if device.type == 'cuda':
                summary['peak_allocated_MiB'] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            print('EXP07_FULL_SMOKE:', json.dumps(summary), flush=True)
        finally:
            sys.path.remove(str(self.source))
            for name in list(sys.modules):
                if name == 'networks' or name.startswith('networks.'):
                    del sys.modules[name]
            sys.modules.update(previous)


if __name__ == '__main__':
    unittest.main()
