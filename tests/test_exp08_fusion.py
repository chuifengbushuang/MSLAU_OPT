"""Fusion-only ablation: initialization, identity-frequency oracle and integration.

Optional 3090 synthetic check: EXP08_FULL_SMOKE=1 EXP08_DEVICE=cuda EXP08_SMOKE_BATCH=12
python -m unittest discover -s tests -p 'test_exp08_fusion.py' -v
"""
import copy
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

from test_exp07_naf import definitions, load

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None

if os.environ.get('EXP08_FULL_SMOKE') == '1' and torch is None:
    raise ImportError('The requested full experiment-8 check requires PyTorch')

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/exp00_baseline'
EXPERIMENT = 'exp08_cascade_fusion'


@unittest.skipIf(torch is None, 'torch is required for fusion numerical checks')
class FusionOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix='exp08_test_')
        cls.source = Path(cls.temporary.name) / 'exp08'
        reference = Path(cls.temporary.name) / 'exp07'
        builder = load('exp08_builder', ROOT / 'tools/build_experiment.py')
        cls.manifest = builder.build(EXPERIMENT, cls.source)
        builder.build('exp07_naf_cascade', reference)
        cls.fusion = load('exp08_fusion', cls.source / 'networks/cascade_fusion.py')
        cls.naf = load('exp08_reference_naf', reference / 'networks/naf_cascade.py')
        names = ['BasicConv2d', 'RFBModified', 'SAM2UNetUp', 'SAM2UNetDecoder']
        cls.original = definitions(BASE / 'networks/mslau_net.py', names, dict(torch=torch, nn=nn, F=F))
        cls.rendered = definitions(cls.source / 'networks/mslau_net.py', names,
            dict(torch=torch, nn=nn, F=F, CascadeFusion2D=cls.fusion.CascadeFusion2D))
        cls.reference_decoder = definitions(reference / 'networks/mslau_net.py', names,
            dict(torch=torch, nn=nn, F=F, CascadeNAF2D=cls.naf.CascadeNAF2D))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_config_and_exported_runtime_are_independent(self):
        actual = json.loads((self.source / 'config.json').read_text())
        reference = json.loads((ROOT / 'experiments/exp07_naf_cascade/config.json').read_text())
        reference['id'] = EXPERIMENT
        reference['train_options']['naf_reg_weight'] = 0.0
        self.assertEqual(actual, reference)
        self.assertEqual(set(self.manifest['added_files']), {'networks/cascade_fusion.py'})
        self.assertFalse((self.source / 'networks/naf_cascade.py').exists())
        launcher = load('exp08_launcher', self.source / 'run.py')
        train, evaluate = launcher.commands(1234, 'preview')
        self.assertEqual(train[train.index('--naf_reg_weight') + 1], '0.0')
        self.assertNotIn('--naf_reg_weight', evaluate)
        self.assertTrue(evaluate[evaluate.index('--model') + 1].endswith('/kvasir_best_model.pth'))

    def test_shared_initial_parameters_and_rng_match_experiment7(self):
        for seed in (1234, 42, 2026):
            for channels in (8, 64):
                torch.manual_seed(seed)
                reference = self.naf.CascadeNAF2D(channels)
                rng = torch.get_rng_state()
                torch.manual_seed(seed)
                model = self.fusion.CascadeFusion2D(channels)
                torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
                reference_state = reference.state_dict()
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, reference_state[name], atol=0, rtol=0)
                removed = set(reference_state) - set(model.state_dict())
                self.assertTrue(all('.selector.' in name or name.endswith('.alpha_logit') for name in removed))
                self.assertEqual(len(removed), 3 * 5)
                if channels == 64:
                    self.assertEqual(sum(p.numel() for p in model.parameters()), 55728)
                    self.assertEqual(sum(p.numel() for p in reference.parameters()), 56124)

    def test_matches_identity_filtered_exp7_after_nonzero_weights_and_in_backward(self):
        torch.manual_seed(31)
        reference = self.naf.DualInputNAF2D(2, 8)
        # The comparison must exercise the learned route, not just zero output.
        nn.init.normal_(reference.output_projection.weight, std=0.1)
        nn.init.normal_(reference.fusion_gate[-1].weight, std=0.1)
        model = self.fusion.DualInputFusion2D(2, 8)
        model.load_state_dict({k: v for k, v in reference.state_dict().items()
                               if k in model.state_dict()}, strict=True)
        a = torch.randn(2, 8, 22, 22, requires_grad=True)
        c = torch.randn(2, 8, 11, 11, requires_grad=True)
        a2, c2 = a.detach().clone().requires_grad_(), c.detach().clone().requires_grad_()

        def identity_filter(local, context):
            return local, context, local.new_zeros(2), local.new_zeros(2, 8)

        # A test-only oracle: replace ONLY exp7's filters by identity. The new
        # production implementation contains no patching or dependency on exp7.
        with patch.object(reference, 'filter_pair', side_effect=identity_filter), \
             patch('torch.fft.rfft2', side_effect=AssertionError('FFT must not execute')), \
             patch('torch.fft.irfft2', side_effect=AssertionError('IFFT must not execute')):
            expected = reference(a, c)
            actual = model(a2, c2)
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected[3], atol=0, rtol=0)
        expected[0].square().mean().backward()
        actual[0].square().mean().backward()
        torch.testing.assert_close(a.grad, a2.grad, atol=0, rtol=0)
        torch.testing.assert_close(c.grad, c2.grad, atol=0, rtol=0)
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, atol=0, rtol=0)

    def test_cascade_has_correct_context_and_cross_level_gradient_without_fft(self):
        model = self.fusion.CascadeFusion2D(4)
        features = [torch.randn(2, 4, h, h, requires_grad=True) for h in (88, 44, 22, 11)]
        records, order, handles = {}, [], []
        for level in (1, 2, 3):
            block = model.stages['p%d' % level]
            nn.init.normal_(block.output_projection.weight, std=0.1)

            def record(module, inputs, output, level=level):
                records[level] = (inputs, output)
                order.append(level)

            handles.append(block.register_forward_hook(record))
        try:
            with patch('torch.fft.rfft2', side_effect=AssertionError('FFT must not execute')), \
                 patch('torch.fft.irfft2', side_effect=AssertionError('IFFT must not execute')):
                enhanced, stats = model(features)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(order, [3, 2, 1])
        self.assertEqual(set(model.stages), {'p1', 'p2', 'p3'})
        self.assertIs(records[3][0][1], features[3])
        self.assertIs(records[2][0][1], records[3][1][0])
        self.assertIs(records[1][0][1], records[2][1][0])
        self.assertIs(enhanced[3], features[3])
        self.assertEqual(stats.shape, (3, 4))
        enhanced[0].square().mean().backward()
        self.assertGreater(features[3].grad.abs().max().item(), 0)
        self.assertGreater(model.stages['p3'].output_projection.weight.grad.abs().max().item(), 0)

    def test_decoder_original_parameters_initial_outputs_and_rng_match(self):
        options = dict(encoder_channels=(4, 8, 16, 32), channels=8)
        for seed in (1234, 42, 2026):
            torch.manual_seed(seed)
            original = self.original['SAM2UNetDecoder'](**options)
            rng = torch.get_rng_state()
            torch.manual_seed(seed)
            reference = self.reference_decoder['SAM2UNetDecoder'](**options)
            torch.manual_seed(seed)
            model = self.rendered['SAM2UNetDecoder'](**options)
            torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
            inputs = [torch.randn(2, c, h, h) for c, h in zip((4, 8, 16, 32), (88, 44, 22, 11))]
            for training in (True, False):
                original.train(training)
                model.train(training)
                with torch.no_grad():
                    expected, actual = original(inputs, 352, True), model(inputs, 352, True)
                self.assertNotIn('naf_regularization', actual)
                for key in ('logits', 'aux_d2', 'aux_d3'):
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_gates_wake_up_and_strict_checkpoint_roundtrip(self):
        model = self.fusion.DualInputFusion2D(2, 8)
        local, context = torch.randn(2, 8, 22, 22), torch.randn(2, 8, 11, 11)
        torch.testing.assert_close(model(local, context)[0], local, atol=0, rtol=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(3):
            optimizer.zero_grad()
            model(local, context)[0].square().mean().backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if step == 2:
                    self.assertGreater(parameter.grad.abs().max().item(), 0, name)
            optimizer.step()
        self.assertGreater((model(local, context)[0] - model(local, -context)[0]).abs().max().item(), 1e-5)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = self.fusion.DualInputFusion2D(2, 8)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(restored(local, context)[0], model(local, context)[0], atol=0, rtol=0)
        with self.assertRaises(RuntimeError):
            self.fusion.DualInputFusion2D(2, 8).load_state_dict(
                self.naf.DualInputNAF2D(2, 8).state_dict(), strict=True)

    def test_real64_channel_pyramid_autocast_backward(self):
        model = self.fusion.CascadeFusion2D(64)
        features = [torch.randn(2, 64, h, h, dtype=torch.bfloat16, requires_grad=True)
                    for h in (88, 44, 22, 11)]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            enhanced, stats = model(features)
            sum(x.float().square().mean() for x in enhanced).backward()
        self.assertTrue(torch.isfinite(stats).all())
        self.assertTrue(all(x.dtype == torch.bfloat16 and torch.isfinite(x.grad).all() for x in features))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_rendered_decoder_optimizer_and_strict_reload(self):
        options = dict(encoder_channels=(4, 8, 16, 32), channels=8)
        model = self.rendered['SAM2UNetDecoder'](**options)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        owned = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in owned for p in model.frequency_enhancer.parameters()))
        inputs = [torch.randn(2, c, h, h) for c, h in zip((4, 8, 16, 32), (88, 44, 22, 11))]
        with patch('torch.fft.rfft2', side_effect=AssertionError('FFT must not execute')), \
             patch('torch.fft.irfft2', side_effect=AssertionError('IFFT must not execute')):
            output = model(inputs, 352, True)
            sum(output[k].square().mean() for k in ('logits', 'aux_d2', 'aux_d3')).backward()
            optimizer.step()
            restored = self.rendered['SAM2UNetDecoder'](**options)
            restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
            model.eval()
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(inputs, 352), restored(inputs, 352), atol=0, rtol=0)

    def test_actual_training_no_regularizer_and_three_checkpoints(self):
        namespace = dict(torch=torch, nn=nn, F=F, copy=copy, os=os, time=time, logging=logging,
                         Variable=torch.autograd.Variable, FUSION_STATS_NAMES=self.fusion.FUSION_STATS_NAMES,
                         log_fusion_scale_stats=lambda *args: None)
        definitions(self.source / 'kvasir_train.py',
                    ['snapshot_state_dict', 'atomic_save_state_dict', 'train_model', 'make_boundary_target'], namespace)
        fusion_type = self.fusion.CascadeFusion2D

        class TinyModel(nn.Module):
            decoder_mode = 'p5_sam2unet'

            def __init__(self):
                super().__init__()
                self.progressive_decoder = nn.Module()
                self.progressive_decoder.frequency_enhancer = fusion_type(1)

            def forward(self, x, **kwargs):
                features, stats = self.progressive_decoder.frequency_enhancer([x] * 4)
                return dict(logits=features[0], fusion_stats=stats)

        inputs = torch.randn(2, 1, 11, 11)
        batches = [(inputs, torch.zeros_like(inputs), ['a', 'b'])]
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(logging, 'info') as log, \
             patch.object(torch.cuda, 'is_available', return_value=False):
            namespace['train_model'](model, nn.BCEWithLogitsLoss(), optimizer, scheduler,
                dict(train=batches, valid=batches), lambda x, y: x.new_tensor(0.5), 1, tmp, boundary_weight=0)
            files = {p.name for p in Path(tmp).glob('*.pth')}
            self.assertEqual(len(files), 3)
            self.assertIn('kvasir_best_model.pth', files)
            self.assertTrue(any(n.startswith('best_loss_') for n in files))
            self.assertTrue(any(n.startswith('best_iou_') for n in files))
            lines = [call.args[0] for call in log.call_args_list if call.args]
            train = next(line for line in lines if line.startswith('train Loss:'))
            valid = next(line for line in lines if line.startswith('valid Loss:'))
            self.assertEqual(train.split()[2], valid.split()[2])
            self.assertEqual(train.split()[4], valid.split()[4])
            self.assertEqual(sum(line == '%s Fusion%d: %s' for line in lines), 6)

    @unittest.skipUnless(os.environ.get('EXP08_FULL_SMOKE') == '1', 'optional real Hiera smoke not requested')
    def test_full_pretrained_hiera_batch12_forward_backward(self):
        checkpoint = os.environ.get('HIERA_CHECKPOINT',
            '/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors')
        self.assertTrue(Path(checkpoint).is_file(), checkpoint)
        device = torch.device(os.environ.get('EXP08_DEVICE', 'cuda'))
        batch = int(os.environ.get('EXP08_SMOKE_BATCH', '12'))
        self.assertGreater(batch, 0)
        previous = {name: sys.modules.pop(name) for name in list(sys.modules)
                    if name == 'networks' or name.startswith('networks.')}
        sys.path.insert(0, str(self.source))
        try:
            from networks.mslau_net import MSLAU_net
            from loss import StructureLoss_binary
            model = MSLAU_net(img_size=352, decoder_mode='p5_sam2unet',
                              encoder_name='sam2_hiera_large', hiera_checkpoint=checkpoint).to(device)
            self.assertTrue(model.encoder.freeze_backbone)
            trainable = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
            self.assertTrue(trainable and all('.prompt_learn.' in n for n in trainable))
            groups = [[p for p in model.encoder.parameters() if p.requires_grad],
                      [p for n, p in model.named_parameters() if not n.startswith('encoder.')]]
            optimizer = torch.optim.AdamW([dict(params=p, lr=1e-3) for p in groups], weight_decay=5e-4)
            enhancer = model.progressive_decoder.frequency_enhancer
            owned = {id(p) for group in optimizer.param_groups for p in group['params']}
            self.assertEqual(sum(p.numel() for p in enhancer.parameters()), 55728)
            self.assertTrue(all(id(p) in owned for p in enhancer.parameters()))
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            output = model(torch.randn(batch, 3, 352, 352, device=device), return_aux=True)
            self.assertNotIn('naf_regularization', output)
            target = torch.randint(0, 2, (batch, 1, 352, 352), device=device).float()
            criterion = StructureLoss_binary()
            loss = sum(criterion(output[key], target) for key in ('logits', 'aux_d2', 'aux_d3'))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for name, parameter in enhancer.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for block in enhancer.stages.values():
                self.assertGreater(block.output_projection.weight.grad.abs().max().item(), 0)
            result = dict(batch=batch, device=str(device), added_parameters=55728,
                          frequency_processing='bypassed', naf_reg_weight=0.0)
            if device.type == 'cuda':
                result['peak_allocated_MiB'] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            print('EXP08_FULL_SMOKE:', json.dumps(result), flush=True)
        finally:
            sys.path.remove(str(self.source))
            for name in list(sys.modules):
                if name == 'networks' or name.startswith('networks.'):
                    del sys.modules[name]
            sys.modules.update(previous)


if __name__ == '__main__':
    unittest.main()
