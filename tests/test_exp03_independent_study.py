"""Current-server integration of the unchanged experiment3 model."""
import argparse
import sys
import tempfile
import types
from pathlib import Path
import unittest

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import exp03_independent_study as driver
import local_decoder_study as common


class IndependentStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_input_audit_allows_only_the_known_initial_hash_logging_assignment(self):
        from run_exp03_independent_study import paired_source_compatible
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);current=root/'current';baseline=root/'baseline'
            common.build(driver.EXPERIMENT,current)
            common.build('exp00_baseline_paired_strict_v2',baseline)
            a,b=current/'paired_inputs.py',baseline/'paired_inputs.py'
            self.assertNotEqual(a.read_bytes(),b.read_bytes())
            self.assertTrue(paired_source_compatible(a,b))
            altered=root/'altered.py'
            altered.write_text(a.read_text().replace('paired_aug_v2','paired_aug_v3'))
            self.assertFalse(paired_source_compatible(altered,b))
            altered.write_text(a.read_text()+"\ndef new_input_behavior():\n    return 1\n")
            self.assertFalse(paired_source_compatible(altered,b))
            altered.write_text(a.read_text().replace("record['baseline_state_sha256']", "record['other_hash']"))
            self.assertFalse(paired_source_compatible(altered,b))

    def test_factory_isolates_cached_baseline_imports_and_restores_modules(self):
        prior={n:sys.modules[n] for n in list(sys.modules) if n=='networks' or n.startswith('networks.')}
        sentinel=types.ModuleType('networks')
        sentinel.__path__=[]
        cached=types.ModuleType('networks.mslau_net')
        sys.modules['networks']=sentinel;sys.modules['networks.mslau_net']=cached
        try:
            with tempfile.TemporaryDirectory() as tmp:
                source=Path(tmp);(source/'networks').mkdir();(source/'networks/__init__.py').write_text('')
                (source/'networks/mslau_net.py').write_text(
                    'from torch import nn\nclass MSLAU_net(nn.Module):\n'
                    '    def __init__(self,**kwargs):\n        super().__init__()\n        self.arguments=kwargs\n')
                model=driver.make_model(source,argparse.Namespace(hiera_checkpoint=Path('/tmp/unused.safetensors')))
                self.assertEqual(model.arguments,dict(img_size=352,decoder_mode='p5_sam2unet',
                    encoder_name='sam2_hiera_large',hiera_checkpoint='/tmp/unused.safetensors',naf_stages=(1,2)))
                self.assertIs(sys.modules['networks'],sentinel)
                self.assertIs(sys.modules['networks.mslau_net'],cached)
                # Exception handling must restore cached reference model imports.
                with self.assertRaises(ModuleNotFoundError):
                    driver.make_model(source/'missing',argparse.Namespace(hiera_checkpoint='unused'))
                self.assertIs(sys.modules['networks'],sentinel)
                self.assertIs(sys.modules['networks.mslau_net'],cached)
        finally:
            for n in list(sys.modules):
                if n=='networks' or n.startswith('networks.'):del sys.modules[n]
            sys.modules.update(prior)

    def test_real_original_decoder_outputs_loss_and_gradient_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';common.build(driver.EXPERIMENT,source)
            self.assertEqual((source/'networks/naf_select_reg.py').read_bytes(),driver.CORE.read_bytes())
            naf=common.load_module('independent_test_original_naf',source/'networks/naf_select_reg.py')
            classes=common.definitions(source/'networks/mslau_net.py',
                {'BasicConv2d','RFBModified','SAM2UNetUp','SAM2UNetDecoder'},
                dict(torch=torch,nn=nn,F=torch.nn.functional,PyramidNAF2D=naf.PyramidNAF2D))
            decoder_type=classes['SAM2UNetDecoder']
            class Tiny(nn.Module):
                def __init__(self):
                    super().__init__();self.encoder=nn.Module()
                    self.encoder.prompt_learn=nn.Conv2d(3,4,1)
                    self.encoder.levels=nn.ModuleList([nn.Conv2d(4,c,1) for c in (4,8,16,32)])
                    self.progressive_decoder=decoder_type((4,8,16,32),4,naf_stages=(1,2))
                def forward(self,x):
                    adapted=self.encoder.prompt_learn(x)
                    xs=[layer(torch.nn.functional.interpolate(adapted,size=(h,h)))
                        for layer,h in zip(self.encoder.levels,(32,16,8,4))]
                    return self.progressive_decoder(xs,32,True)
            model=Tiny();optimizer=common.make_optimizer(model)
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                outputs=model(torch.randn(2,3,32,32))
                _,loss=common.losses(outputs,torch.zeros(2,1,32,32),nn.MSELoss(),'exp03_independent',True)
                loss.backward()
                if step==2:driver.check_gradients(driver.gradient_report(model),'exp03_independent')
                optimizer.step()
            enhancer=model.progressive_decoder.frequency_enhancer
            self.assertEqual(enhancer.active_stages,(1,2))
            self.assertEqual(sum(p.numel() for p in enhancer.parameters()),164)
            self.assertFalse(hasattr(enhancer,'set_selection_context'))

    def test_callbacks_preserve_original_regularizer_and_do_not_add_context(self):
        names=('gradient_report','check_gradients','set_context')
        previous={n:getattr(common,n) for n in names};losses=common.losses
        try:
            driver.install_loop_hooks()
            self.assertIs(common.losses,losses)
            self.assertIs(common.set_context,driver.no_selection_context)
            self.assertIsNone(common.set_context(None,None,0,('image',)))
            for training in (True,False):
                outputs={n:torch.randn(2,1,4,4,requires_grad=True) for n in ('logits','aux_d2','aux_d3')}
                penalty=torch.tensor(1.,requires_grad=True);outputs['naf_regularization']=penalty
                _,loss=common.losses(outputs,torch.zeros(2,1,4,4),nn.MSELoss(),'exp03_independent',training)
                loss.backward()
                if training:self.assertAlmostEqual(penalty.grad.item(),.001,places=8)
                else:self.assertIsNone(penalty.grad)
        finally:
            for n,v in previous.items():setattr(common,n,v)


if __name__=='__main__':unittest.main()
