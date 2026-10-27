"""Actual decoder replacement, shared initialization, gradients and serialization."""
import importlib.util
import io
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import local_decoder_study as study


class LocalDecoderStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        names = {'BasicConv2d', 'RFBModified', 'SAM2UNetUp', 'SAM2UNetDecoder'}
        cls.original = study.definitions(ROOT / 'experiments/exp00_baseline/networks/mslau_net.py',
            names, dict(torch=torch, nn=nn, F=F))['SAM2UNetDecoder']
        cls.naf = study.load_module('study_test_naf', study.NAF_FILE)

    def decoder(self, replacement=False):
        original = self.original((4, 8, 16, 32), 4)
        return study.StudyDecoder(original, self.naf.CascadeDiscreteNAF2D(4), replacement)

    def features(self, requires_grad=False):
        return [torch.randn(2, c, h, h, requires_grad=requires_grad)
                for c, h in zip((4, 8, 16, 32), (32, 16, 8, 4))]

    def test_shared_layers_rng_and_zero_projection_parity(self):
        torch.manual_seed(1234)
        original = self.original((4, 8, 16, 32), 4).eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(1234)
        appended = self.decoder().eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, appended.state_dict()[key]), key)
        x = self.features()
        torch.testing.assert_close(original(x, 32), appended(x, 32), atol=0, rtol=0)
        torch.manual_seed(1234)
        replaced = self.decoder(True).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in replaced.state_dict().items():
            if not key.startswith('frequency_enhancer.'):
                self.assertTrue(torch.equal(value, original.state_dict()[key]), key)

    def test_replacement_removes_ups_and_heads_read_actual_naf_features(self):
        decoder = self.decoder(True).eval()
        self.assertFalse(any(key.startswith(('up1.', 'up2.', 'up3.')) for key in decoder.state_dict()))
        for block in decoder.frequency_enhancer.stages.values():
            nn.init.normal_(block.output_projection.weight, std=.1)
        x = self.features()
        with torch.no_grad():
            features = [rfb(t) for rfb, t in zip(decoder.rfbs, x)]
            enhanced = decoder.frequency_enhancer(features)[0]
            actual = decoder(x, 32, return_aux=True)
            expected = dict(logits=decoder.head(enhanced[0]), aux_d2=decoder.side2(enhanced[1]),
                            aux_d3=decoder.side1(enhanced[2]))
            for name, value in expected.items():
                torch.testing.assert_close(actual[name], decoder.resize_logits(value, 32), atol=0, rtol=0)

    def test_task_gradient_opens_all_three_stages_and_all_original_inputs(self):
        for replacement in (False, True):
            decoder = self.decoder(replacement).train()
            decoder.frequency_enhancer.set_selection_context(1234, 0, ['a', 'b'])
            optimizer = torch.optim.AdamW(decoder.parameters(), lr=.001)
            for _ in range(3):
                x = self.features(True)
                optimizer.zero_grad(set_to_none=True)
                output = decoder(x, 32, return_aux=True)
                task = sum(output[k].square().mean() for k in ('logits', 'aux_d2', 'aux_d3'))
                # Pure task loss verifies gradients beyond regularizer-only paths.
                task.backward()
                optimizer.step()
            for level, block in decoder.frequency_enhancer.stages.items():
                for name in ('selector', 'fusion_gate', 'output_projection'):
                    gradients = [p.grad for p in getattr(block, name).parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, (replacement, level, name))
            for value in x:
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_full_checkpoint_strict_reload_and_deterministic_inference(self):
        for replacement in (False, True):
            decoder = self.decoder(replacement).eval()
            for block in decoder.frequency_enhancer.stages.values():
                nn.init.normal_(block.output_projection.weight, std=.1)
            x = self.features()
            with torch.no_grad():
                expected = decoder(x, 32)
                buffer = io.BytesIO()
                torch.save(decoder.state_dict(), buffer)
                buffer.seek(0)
                restored = self.decoder(replacement).eval()
                restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                torch.rand(17)
                torch.testing.assert_close(expected, restored(x, 32), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
