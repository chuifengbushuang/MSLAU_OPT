"""Residual-source controls, unchanged frequency selection and trainable paths."""
import io
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import local_decoder_study as common
import sgg_residual_study as driver

core = common.load_module('test_sgg_residual_core', driver.CORE)
legacy = common.load_module('test_sgg_legacy', common.NAF_FILE)


class ResidualGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, mode):
        return core.install_residual_gates(legacy.CascadeDiscreteNAF2D(4), mode)

    def test_initial_legacy_parameters_rng_and_nonzero_residual_parity(self):
        for seed in (1234, 42, 2026):
            torch.manual_seed(seed)
            original = self.make('original').double().eval()
            before = torch.get_rng_state().clone()
            local, deep = self.make('local').double().eval(), self.make('deep').double().eval()
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(list(local.state_dict()), list(deep.state_dict()))
            for k, v in local.state_dict().items():
                self.assertTrue(torch.equal(v, deep.state_dict()[k]), k)
                if '.residual_gate.' not in k:
                    self.assertTrue(torch.equal(v, original.state_dict()[k]), k)
            # Opening the original output projection proves parity beyond its
            # trivial initial zero residual, including frequency stats/penalty.
            for stage in original.stages.values():
                nn.init.normal_(stage.output_projection.weight, std=.1)
            for model in (local, deep):
                model.load_state_dict(original.state_dict(), strict=False)
            features = [torch.randn(2, 4, h, h, dtype=torch.double) for h in (16, 8, 4, 2)]
            expected = original(features)
            for model in (local, deep):
                actual = model(features)
                for a, b in zip(expected[0], actual[0]):
                    self.assertTrue(torch.equal(a, b))
                for a, b in zip(expected[1:], actual[1:]):
                    self.assertTrue(torch.equal(a, b))

    def test_local_vs_deep_source_and_only_raw_local_skip_is_gated(self):
        torch.manual_seed(42)
        local, deep = self.make('local').eval().double(), self.make('deep').eval().double()
        for model in (local, deep):
            nn.init.normal_(model.stages['p1'].residual_gate.mask.weight, std=.1)
        deep.load_state_dict(local.state_dict(), strict=True)
        x = torch.randn(2, 4, 12, 10, dtype=torch.double)
        c = torch.randn(2, 4, 6, 5, dtype=torch.double)
        a, b = local.stages['p1'], deep.stages['p1']
        ya, yb = a(x,c)[0], b(x,c)[0]
        self.assertTrue(torch.equal(ya, a(x,c+2)[0]))
        self.assertFalse(torch.equal(yb, b(x,c+2)[0]))
        self.assertTrue(torch.equal(ya, a.residual_gate(x)*x))
        self.assertTrue(torch.equal(yb, b.residual_gate(torch.nn.functional.interpolate(
            c, size=x.shape[-2:], mode='bilinear', align_corners=True))*x))

    def test_training_selection_keys_and_stats_unchanged_at_unit_gate(self):
        torch.manual_seed(1234)
        original, deep = self.make('original'), self.make('deep')
        xs = [torch.randn(2,4,h,h) for h in (16,8,4,2)]
        for model in (original,deep):
            model.set_selection_context(1234,3,('a','b'))
        rng = torch.get_rng_state().clone()
        a, b = original(xs), deep(xs)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for v,w in zip(a[0],b[0]):self.assertTrue(torch.equal(v,w))
        for v,w in zip(a[1:],b[1:]):self.assertTrue(torch.equal(v,w))
        deep.set_selection_context(1234,4,('a','b'))
        self.assertFalse(torch.equal(deep(xs)[2], b[2]))

    def test_gate_and_original_paths_open_and_repeat_strict_reload(self):
        for mode in ('original','local','deep'):
            trajectories = []
            for repeat in range(2):
                torch.manual_seed(42)
                model = self.make(mode)
                optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
                for step in range(4):
                    xs = [torch.randn(2,4,h,h,requires_grad=True) for h in (16,8,4,2)]
                    model.set_selection_context(42,step,('a','b'))
                    optimizer.zero_grad(set_to_none=True)
                    outputs, penalty, _, _ = model(xs)
                    (sum(v.square().mean() for v in outputs[:3])+.001*penalty).backward()
                    if step == 0 and mode != 'original':
                        for stage in model.stages.values():
                            self.assertGreater(stage.residual_gate.mask.weight.grad.abs().max().item(),0)
                            self.assertEqual(stage.residual_gate.standard.weight.grad.abs().max().item(),0)
                    optimizer.step()
                for name,p in model.named_parameters():
                    self.assertIsNotNone(p.grad,name)
                    self.assertTrue(torch.isfinite(p.grad).all(),name)
                    self.assertGreater(p.grad.abs().max().item(),0,name)
                self.assertTrue(all(x.grad is not None and x.grad.abs().max()>0 for x in xs))
                model.eval()
                data = io.BytesIO();torch.save(model.state_dict(),data);data.seek(0)
                restored = self.make(mode).eval()
                restored.load_state_dict(torch.load(data,weights_only=True),strict=True)
                for a,b in zip(model(xs)[0],restored(xs)[0]):self.assertTrue(torch.equal(a,b))
                trajectories.append({k:v.clone() for k,v in model.state_dict().items()})
            self.assertTrue(all(torch.equal(v,trajectories[1][k]) for k,v in trajectories[0].items()))

    def test_training_regularizer_and_context_callbacks_are_retained(self):
        outputs = {k:torch.randn(2,1,4,4,requires_grad=True) for k in ('logits','aux_d2','aux_d3')}
        outputs['naf_regularization'] = torch.tensor(1.,requires_grad=True)
        for training in (True,False):
            for v in outputs.values():v.grad=None
            _,loss = common.losses(outputs,torch.zeros(2,1,4,4),nn.MSELoss(),'sgg_deep',training)
            loss.backward()
            if training:self.assertAlmostEqual(outputs['naf_regularization'].grad.item(),.001,places=8)
            else:self.assertIsNone(outputs['naf_regularization'].grad)
        import inspect
        source = inspect.getsource(driver.install_loop_hooks)
        self.assertNotIn('common.losses =',source)
        self.assertNotIn('common.set_context =',source)


if __name__ == '__main__':
    unittest.main()
