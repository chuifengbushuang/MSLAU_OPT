"""Scientific controls, Fourier geometry, and trainable paths of H/G/FS."""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import freqselect_study as driver
import local_decoder_study as common

core = common.load_module('test_freqselect_core', driver.CORE)
legacy = common.load_module('test_freqselect_shell', common.NAF_FILE)


class FreqSelectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, mode, channels=4):
        return core.CascadeSelection(legacy.CascadeDiscreteNAF2D(channels), mode)

    def test_masks_reconstruction_conjugacy_odd_even_rectangular(self):
        for h, w in ((7, 9), (8, 10), (11, 11), (22, 22), (44, 44), (88, 88)):
            x = torch.randn(2, 4, h, w, dtype=torch.double)
            module = core.BandSelection(4, True).double()
            torch.testing.assert_close(sum(module.decompose(x)), x, atol=1e-14, rtol=1e-14)
            torch.testing.assert_close(module(x), x, atol=1e-14, rtol=1e-14)
            # A full-spectrum reference checks real inverse, not merely a
            # telescoping reconstruction that could hide asymmetric masks.
            for cutoff, mask in zip((.25, .125, .0625), core.lowpass_masks(h, w, x.device, x.dtype)):
                full = torch.maximum(torch.fft.fftfreq(h, dtype=x.dtype).abs()[:, None],
                                     torch.fft.fftfreq(w, dtype=x.dtype).abs()[None, :]) < cutoff
                inverse = torch.fft.ifft2(torch.fft.fft2(x, norm='ortho') * full, norm='ortho')
                self.assertLess(inverse.imag.abs().max().item(), 1e-14)
                actual = torch.fft.irfft2(torch.fft.rfft2(x, norm='ortho') * mask, s=(h, w), norm='ortho')
                torch.testing.assert_close(actual, inverse.real, atol=1e-14, rtol=1e-14)

    def test_frequency_selection_has_real_band_effect_and_keeps_dc(self):
        module = core.BandSelection(1, True).double()
        for gate in module.gates:
            gate.bias.data.fill_(torch.log(torch.tensor(1 / 3.)).item())  # weight .5
        constant = torch.ones(2, 1, 32, 32, dtype=torch.double)
        high = ((torch.arange(32) % 2) * 2 - 1).double()[None, None, None].expand_as(constant)
        torch.testing.assert_close(module(constant), constant, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(module(high), .5 * high, atol=2e-8, rtol=2e-8)
        module.gates[0].weight.data[0, 0, 1, 1] = .3
        weights = 2 * module.gates[0](high).sigmoid()
        self.assertGreater(weights.std().item(), 0)
        torch.testing.assert_close(module(high), weights * high, atol=1e-14, rtol=1e-14)

    def test_g_is_exactly_generic_spatial_attention_and_calls_no_fft(self):
        module = core.BandSelection(4, False).double()
        x = torch.randn(2, 4, 7, 9, dtype=torch.double)
        for gate in module.gates:
            nn.init.normal_(gate.weight, std=.1)
        with patch.object(torch.fft, 'rfft2', side_effect=AssertionError('G called FFT')):
            actual = module(x)
        expected = x * (1 + sum(2 * gate(x).sigmoid() for gate in module.gates)) / 4
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)

    def test_parameter_initialization_shared_shell_and_rng(self):
        for seed in (1234, 42, 2026):
            torch.manual_seed(seed)
            reference = legacy.CascadeDiscreteNAF2D(4)
            before = torch.get_rng_state().clone()
            h, g, fs = [self.make(mode) for mode in ('h', 'g', 'fs')]
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(set(g.state_dict()), set(fs.state_dict()))
            for name, value in g.state_dict().items():
                self.assertTrue(torch.equal(value, fs.state_dict()[name]), name)
                if '.filter.' not in name:
                    self.assertTrue(torch.equal(value, h.state_dict()[name]), name)
                    self.assertTrue(torch.equal(value, reference.state_dict()[name]), name)
            self.assertFalse(any('selector' in name for name in fs.state_dict()))
            self.assertEqual(sum(p.numel() for p in g.parameters()) - sum(p.numel() for p in h.parameters()), 9 * (9 * 4 + 1))

    def test_zero_projection_all_levels_unchanged(self):
        features = [torch.randn(2, 4, n, n) for n in (16, 8, 4, 2)]
        for mode in ('h', 'g', 'fs'):
            values = self.make(mode)(features)
            for a, b in zip(features, values[0]):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(values[1].item(), 0)

    def test_task_gradients_open_in_all_three_stages_and_streams(self):
        for mode in ('h', 'g', 'fs'):
            model = self.make(mode)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            for step in range(3):
                features = [torch.randn(2, 4, n, n, requires_grad=True) for n in (44, 22, 11, 6)]
                optimizer.zero_grad(set_to_none=True)
                sum(v.square().mean() for v in model(features)[0][:3]).backward()
                if step == 0:
                    for stage in model.stages.values():
                        for p in stage.filter.parameters():
                            self.assertIsNotNone(p.grad)
                            self.assertEqual(torch.count_nonzero(p.grad).item(), 0)
                optimizer.step()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, (mode, name))
                self.assertTrue(torch.isfinite(p.grad).all(), (mode, name))
                self.assertGreater(p.grad.abs().max().item(), 0, (mode, name))
            self.assertTrue(all(v.grad is not None and v.grad.abs().sum() > 0 for v in features))

    def test_full_serialization_and_cpu_repeat_trajectory(self):
        for mode in ('h', 'g', 'fs'):
            hashes = []
            for _ in range(2):
                torch.manual_seed(42)
                model = self.make(mode)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                for step in range(3):
                    xs = [torch.randn(2, 4, n, n) for n in (16, 8, 4, 2)]
                    optimizer.zero_grad(set_to_none=True)
                    sum(v.square().mean() for v in model(xs)[0][:3]).backward()
                    optimizer.step()
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer);buffer.seek(0)
                restored = self.make(mode)
                restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                for a, b in zip(model(xs)[0], restored(xs)[0]):
                    self.assertTrue(torch.equal(a, b))
                hashes.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            self.assertTrue(all(torch.equal(v, hashes[1][k]) for k, v in hashes[0].items()))

    def test_actual_loop_loss_does_not_use_old_selector_regularization(self):
        outputs = {k: torch.randn(2, 1, 4, 4, requires_grad=True) for k in ('logits', 'aux_d2', 'aux_d3')}
        penalty = torch.tensor(999., requires_grad=True);outputs['naf_regularization'] = penalty
        _, total = driver.losses(outputs, torch.zeros(2, 1, 4, 4), nn.MSELoss(), 'freqselect_fs', True)
        total.backward()
        self.assertIsNone(penalty.grad)
        self.assertTrue(all(outputs[k].grad.abs().sum() > 0 for k in ('logits', 'aux_d2', 'aux_d3')))


if __name__ == '__main__':
    unittest.main()
