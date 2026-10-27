"""Cross-scale topology, zero-beta parity, task gradients and checkpoint reload."""
import copy
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


class CrossScaleSkipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.naf = study.load_module('skip_test_naf', study.NAF_FILE)
        cls.skip = study.load_module('skip_test_module', study.SKIP_FILE)
        cls.original_decoder = study.definitions(ROOT / 'experiments/exp00_baseline/networks/mslau_net.py',
            {'BasicConv2d', 'RFBModified', 'SAM2UNetUp', 'SAM2UNetDecoder'},
            dict(torch=torch, nn=nn, F=F))['SAM2UNetDecoder']

    def enhancer(self):
        return self.skip.CrossScaleSkipNAF2D(self.naf.CascadeDiscreteNAF2D(4))

    def decoder(self):
        return study.StudyDecoder(self.original_decoder((4, 8, 16, 32), 4), self.enhancer(), True)

    def test_zero_beta_preserves_all_original_parameters_rng_and_nonzero_outputs(self):
        torch.manual_seed(1234)
        original = self.naf.CascadeDiscreteNAF2D(4).eval()
        for block in original.stages.values():
            nn.init.normal_(block.output_projection.weight, std=.1)
        state = torch.get_rng_state().clone()
        new = self.skip.CrossScaleSkipNAF2D(copy.deepcopy(original)).eval()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in original.parameters()), 3)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], atol=0, rtol=0)
        features = [torch.randn(2, 4, h, h) for h in (32, 16, 8, 4)]
        for dtype in (torch.float32, torch.bfloat16):
            x = [t.to(dtype) for t in features]
            a, b = original(x), new(x)
            for before, after in zip(a[0], b[0]):
                torch.testing.assert_close(before, after, atol=0, rtol=0)
            for before, after in zip(a[1:], b[1:]):
                torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_nonzero_skip_is_propagated_into_the_next_naf_and_all_heads(self):
        decoder = self.decoder().eval()
        enhancer = decoder.frequency_enhancer
        for level, beta in ((1, .3), (2, -.2), (3, .1)):
            enhancer.betas[f'p{level}'].data.fill_(beta)
            nn.init.normal_(enhancer.stages[f'p{level}'].output_projection.weight, std=.1)
        x = [torch.randn(2, c, h, h) for c, h in zip((4, 8, 16, 32), (32, 16, 8, 4))]
        with torch.no_grad():
            features = [rfb(t) for rfb, t in zip(decoder.rfbs, x)]
            context, expected = features[3], {}
            for level in (3, 2, 1):
                local = features[level - 1]
                filtered = enhancer.stages[f'p{level}'](local, context)[0]
                context = filtered + enhancer.betas[f'p{level}'] * F.interpolate(
                    context, size=local.shape[-2:], mode='bilinear', align_corners=True)
                expected[level] = context
            actual = decoder(x, 64, return_aux=True)
            for key, head, level in (('logits', decoder.head, 1), ('aux_d2', decoder.side2, 2), ('aux_d3', decoder.side1, 3)):
                torch.testing.assert_close(actual[key], decoder.resize_logits(head(expected[level]), 64), atol=0, rtol=0)
        self.assertFalse(any(key.startswith(('up1.', 'up2.', 'up3.')) for key in decoder.state_dict()))

    def test_zero_beta_has_task_gradient_and_all_frequency_branches_open(self):
        torch.manual_seed(42)
        decoder = self.decoder().train()
        decoder.frequency_enhancer.set_selection_context(1234, 0, ['a', 'b'])
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=.001)
        for step in range(3):
            x = [torch.randn(2, c, h, h, requires_grad=True)
                 for c, h in zip((4, 8, 16, 32), (32, 16, 8, 4))]
            optimizer.zero_grad(set_to_none=True)
            output = decoder(x, 64, return_aux=True)
            sum(output[key].square().mean() for key in ('logits', 'aux_d2', 'aux_d3')).backward()
            for beta in decoder.frequency_enhancer.betas.values():
                self.assertIsNotNone(beta.grad)
                self.assertTrue(torch.isfinite(beta.grad))
                self.assertGreater(beta.grad.abs().item(), 0)
                if step == 0:
                    self.assertEqual(beta.item(), 0)
            optimizer.step()
        for beta in decoder.frequency_enhancer.betas.values():
            self.assertNotEqual(beta.item(), 0)
        for block in decoder.frequency_enhancer.stages.values():
            for name in ('selector', 'fusion_gate', 'output_projection'):
                gradients = [p.grad for p in getattr(block, name).parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        for value in x:
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_nonzero_beta_full_checkpoint_strict_reload_and_deterministic_eval(self):
        decoder = self.decoder().eval()
        for beta in decoder.frequency_enhancer.betas.values():
            beta.data.fill_(.15)
        x = [torch.randn(2, c, h, h) for c, h in zip((4, 8, 16, 32), (32, 16, 8, 4))]
        with torch.no_grad():
            expected = decoder(x, 64)
            buffer = io.BytesIO()
            torch.save(decoder.state_dict(), buffer)
            buffer.seek(0)
            restored = self.decoder().eval()
            restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
            torch.rand(17)
            torch.testing.assert_close(expected, restored(x, 64), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
