"""Check the actual scientific contrast, gradients, and legacy-shell preservation."""
import importlib.util
import io
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import fredft_study as driver
import local_decoder_study as common

core = common.load_module('test_fredft_core', driver.CORE)
legacy = common.load_module('test_fredft_legacy', common.NAF_FILE)


class FreDFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, spectral, channels=4):
        return core.CascadeFreDFT(legacy.CascadeDiscreteNAF2D(channels), spectral)

    def test_shared_state_rng_and_zero_projection(self):
        for seed in (1234, 42, 2026):
            torch.manual_seed(seed)
            reference = legacy.CascadeDiscreteNAF2D(4)
            before = torch.get_rng_state().clone()
            left, right = self.make(True), self.make(False)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            for name, value in left.state_dict().items():
                self.assertTrue(torch.equal(value, right.state_dict()[name]), name)
                if not any(f'.{key}.' in name for key in ('mfda', 'fdffl', 'norm_ffn')):
                    self.assertTrue(torch.equal(value, reference.state_dict()[name]), name)
            self.assertFalse(any('selector' in name for name in left.state_dict()))
            features = [torch.randn(2, 4, n, n) for n in (16, 8, 4, 2)]
            for model in (left, right):
                output = model(features)
                for x, y in zip(features, output[0]):
                    self.assertTrue(torch.equal(x, y))
                self.assertEqual(output[1].item(), 0)

    def test_mfda_same_stream_qk_cross_stream_v_matches_reference(self):
        for spectral in (True, False):
            module = core.MFDA(4, spectral).double()
            x, y = torch.randn(2, 4, 7, 8, dtype=torch.double), torch.randn(2, 4, 7, 8, dtype=torch.double)
            q1, k1, v1 = module.to_hidden_dw(module.to_hidden(module.norm1(x))).chunk(3, 1)
            q2, k2, v2 = module.to_hidden_dw(module.to_hidden(module.norm1(y))).chunk(3, 1)
            response = lambda q, k: torch.fft.irfft2(torch.fft.rfft2(q) * torch.fft.rfft2(k), s=q.shape[-2:]) if spectral else q * k
            expected = (x + module.project_out(module.norm(response(q1, k1)) * v2),
                        y + module.project_out(module.norm(response(q2, k2)) * v1))
            for actual, target in zip(module(x, y), expected):
                torch.testing.assert_close(actual, target, atol=0, rtol=0)

    def test_fdffl_fourier_channel_mix_equals_spatial_including_gradients(self):
        for height, width in ((7, 9), (8, 8), (22, 22), (44, 44), (88, 88)):
            left, right = core.FDFFL(4, True), core.FDFFL(4, False)
            right.load_state_dict(left.state_dict(), strict=True)
            x = torch.randn(2, 4, height, width, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            a, b = left(x), right(y)
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
            a.square().mean().backward()
            b.square().mean().backward()
            torch.testing.assert_close(x.grad, y.grad, atol=2e-9, rtol=1e-4)
            for p, q in zip(left.parameters(), right.parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=2e-8, rtol=2e-4)

    def test_cascade_has_task_gradients_after_output_projection_opens(self):
        for spectral in (True, False):
            model = self.make(spectral)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            for step in range(3):
                features = [torch.randn(2, 4, n, n, requires_grad=True) for n in (16, 8, 4, 2)]
                optimizer.zero_grad(set_to_none=True)
                outputs = model(features)[0]
                sum(x.square().mean() for x in outputs[:3]).backward()
                for stage in model.stages.values():
                    self.assertGreater(stage.output_projection.weight.grad.abs().max().item(), 0)
                    if step == 0:
                        self.assertTrue(all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in stage.mfda.parameters()))
                optimizer.step()
            for stage in model.stages.values():
                for key in ('mfda', 'fdffl', 'fusion_gate', 'local_projection', 'context_projection'):
                    grads = [p.grad for p in getattr(stage, key).parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads), (spectral, key))
                    self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, (spectral, key))
            self.assertTrue(all(x.grad is not None and x.grad.abs().sum() > 0 for x in features))

    def test_strict_roundtrip_and_spatial_control_never_calls_fft(self):
        from unittest.mock import patch
        for spectral in (True, False):
            model = self.make(spectral)
            for stage in model.stages.values():
                nn.init.normal_(stage.output_projection.weight, std=.1)
            features = [torch.randn(2, 4, n, n) for n in (16, 8, 4, 2)]
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = self.make(spectral)
            restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
            if spectral:
                actual, expected = model(features), restored(features)
            else:
                with patch.object(torch.fft, 'rfft2', side_effect=AssertionError('FFT called')):
                    actual, expected = model(features), restored(features)
            for a, b in zip(actual[0], expected[0]):
                self.assertTrue(torch.equal(a, b))

    def test_real_loop_task_loss_has_no_naf_regularizer(self):
        criterion = nn.MSELoss()
        outputs = {k: torch.randn(2, 1, 4, 4, requires_grad=True) for k in ('logits', 'aux_d2', 'aux_d3')}
        penalty = torch.tensor(999., requires_grad=True)
        outputs['naf_regularization'] = penalty
        labels = torch.randn(2, 1, 4, 4)
        main, total = driver.losses(outputs, labels, criterion, 'fredft_frequency', True)
        total.backward()
        self.assertIsNone(penalty.grad)
        self.assertTrue(all(outputs[k].grad.abs().sum() > 0 for k in ('logits', 'aux_d2', 'aux_d3')))


if __name__ == '__main__':
    unittest.main()
