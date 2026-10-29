"""Independent numerical checks for author-model protocol adapters."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import tempfile
import types
import contextlib
import io
import os
import ast

import torch
from torch.nn import functional as F

DIRECTORY = Path(__file__).resolve().parents[1] / 'studies/paper_models'
sys.path.insert(0, str(DIRECTORY))
from dcnv4_reference import dcnv4_reference
from early_stopping import EarlyStopping


def grid_oracle(value, offset_mask, kh, kw, sh, sw, ph, pw, dh, dw, g, d, scale, remove):
    """Different implementation: PyTorch's grid sampler, one group at a time."""
    n, h, w, _ = value.shape
    oh, ow = offset_mask.shape[1:3]
    k = kh * kw - remove
    groups = []
    for gi in range(g):
        field = value[..., gi*d:(gi+1)*d].permute(0, 3, 1, 2)
        om = offset_mask[..., gi*k*3:(gi+1)*k*3]
        rows, cols = torch.meshgrid(torch.arange(oh, dtype=value.dtype),
                                   torch.arange(ow, dtype=value.dtype), indexing='ij')
        result = field.new_zeros(n, d, oh, ow)
        z = 0
        for i in range(kw):
            for j in range(kh):
                if remove and (i, j) == (kw//2, kh//2):
                    continue
                cx, cy = dw*(kw-1)//2, dh*(kh-1)//2
                px = cols*sw + cx-pw + ((i*dw-cx)+om[..., 2*z])*scale
                py = rows*sh + cy-ph + ((j*dh-cy)+om[..., 2*z+1])*scale
                grid = torch.stack(((px+.5)*2/w-1, (py+.5)*2/h-1), dim=-1)
                sampled = F.grid_sample(field, grid, align_corners=False, padding_mode='zeros')
                result = result + sampled*om[..., 2*k+z].unsqueeze(1)
                z += 1
        groups.append(result)
    return torch.cat(groups, dim=1).permute(0, 2, 3, 1)


class DCNv4Tests(unittest.TestCase):
    def test_native_shallow_relu_is_inactive_with_both_sampling_backends(self):
        import adapters
        from dcnv4_reference import DCNv4Function
        base=adapters.ASSETS/'vendor/eat'
        source=ast.parse((base/'src/eat_model/eat.py').read_text())
        definition=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name=='DFConv')
        def oracle_apply(value,offset,*args):
            return grid_oracle(value,offset,*args[:11],args[12])
        for backend in (DCNv4Function,types.SimpleNamespace(apply=oracle_apply)):
            dcn=adapters.source_module('native_dcn_shallow_test',base/'DCNv4_op/DCNv4/modules/dcnv4.py',
                dict(DCNv4Function=backend),remove_imports=('functions',))
            namespace=dict(torch=torch,nn=torch.nn,Tensor=torch.Tensor,DCNv4=dcn.DCNv4)
            exec(compile(ast.Module(body=[definition],type_ignores=[]),'<pinned native DFConv>','exec'),namespace)
            torch.manual_seed(64)
            layer=namespace['DFConv'](64,32,3,padding=1,offset_scale=1.)
            original={k:v.clone() for k,v in layer.named_parameters()}
            optimizer=torch.optim.AdamW(layer.parameters(),lr=.001,weight_decay=.0005)
            for _ in range(6):
                optimizer.zero_grad(set_to_none=True)
                out=layer(torch.randn(2,64,5,5))
                self.assertEqual(out.count_nonzero().item(),0)
                (out-torch.randn_like(out)).square().mean().backward()
                for p in layer.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertEqual(p.grad.count_nonzero().item(),0)
                optimizer.step()
            # AdamW may decay nonzero weights, but zero masks/BN biases remain
            # exactly zero. Changing only ReLU to GELU enables mask gradients.
            self.assertEqual(layer.conv.offset_mask.weight.count_nonzero().item(),0)
            self.assertEqual(layer.conv.offset_mask.bias.count_nonzero().item(),0)
            layer.load_state_dict({**layer.state_dict(),**original})
            layer.act=torch.nn.GELU()
            optimizer.zero_grad(set_to_none=True)
            out=layer(torch.randn(2,64,5,5))
            (out-torch.randn_like(out)).square().mean().backward()
            k=layer.conv.kernel_size**2
            gradient=layer.conv.offset_mask.weight.grad[:layer.conv.group*3*k].reshape(layer.conv.group,3*k,-1)
            self.assertGreater(gradient[:,2*k:].abs().max().item(),0)
            # Zero initial masks still imply zero coordinate gradient on the
            # first step; nonzero mask updates open that path on later steps.
            self.assertEqual(gradient[:,:2*k].count_nonzero().item(),0)

    def test_forward_and_both_gradients_against_independent_grid_sampler(self):
        torch.manual_seed(82)
        for stride, dilation, scale, remove in [(1,1,1.,0), (2,1,2.5,0), (1,2,.7,1)]:
            with self.subTest(stride=stride, dilation=dilation, scale=scale, remove=remove):
                n,h,w,g,d,kh,kw,ph,pw = 2,5,7,2,3,3,3,2,2
                oh = (h+2*ph-dilation*(kh-1)-1)//stride+1
                ow = (w+2*pw-dilation*(kw-1)-1)//stride+1
                k = kh*kw-remove
                x = torch.randn(n,h,w,g*d,dtype=torch.double,requires_grad=True)
                om = torch.randn(n,oh,ow,((g*k*3+7)//8)*8,dtype=torch.double,requires_grad=True)
                args = (kh,kw,stride,stride,ph,pw,dilation,dilation,g,d,scale)
                actual = dcnv4_reference(x,om,*args,remove_center=remove)
                expected = grid_oracle(x,om,*args,remove)
                torch.testing.assert_close(actual,expected,atol=2e-13,rtol=2e-13)
                weight = torch.randn_like(actual)
                a = torch.autograd.grad((actual*weight).sum(),(x,om),retain_graph=True)
                b = torch.autograd.grad((expected*weight).sum(),(x,om))
                for av,bv in zip(a,b):
                    torch.testing.assert_close(av,bv,atol=1e-12,rtol=1e-12)

    def test_finite_difference(self):
        torch.manual_seed(41)
        x = torch.randn(1,2,2,2,dtype=torch.double,requires_grad=True)
        om = (.31+torch.randn(1,2,2,8,dtype=torch.double)*.1).requires_grad_()
        fn = lambda a,b: dcnv4_reference(a,b,1,1,1,1,0,0,1,1,1,2,1.)
        self.assertTrue(torch.autograd.gradcheck(fn,(x,om),atol=1e-5,rtol=1e-4))

    def test_unnormalized_mask_and_group_separation(self):
        x = torch.arange(16,dtype=torch.double).reshape(1,2,2,4)
        om = torch.zeros(1,2,2,8,dtype=torch.double)
        om[...,2],om[...,5] = 2.,-3.
        y = dcnv4_reference(x,om,1,1,1,1,0,0,1,1,2,2,1.)
        torch.testing.assert_close(y,torch.cat([x[...,:2]*2,x[...,2:]*-3],dim=-1))

    def test_protocol_preserves_user_scope_and_selection(self):
        config = json.loads((DIRECTORY/'protocol.json').read_text())
        self.assertEqual(config['models'],['polyper','eat','condseg','asgnet'])
        self.assertEqual(config['seeds'],[1234,42,2026])
        self.assertEqual((config['epochs'],config['image_size'],config['batch_size']),(100,352,12))
        self.assertEqual(config['early_stopping']['patience'],10)
        self.assertFalse(config['early_stopping']['ties_reset_patience'])
        self.assertIn('same 120',config['checkpoint_selection'])
        self.assertIn('l2s',config['deferred'])

    @unittest.skipUnless(os.environ.get('PAPER_MODELS_CUDA_TESTS')=='1','Explicit CUDA numerical check')
    def test_cuda_strict_backward_is_repeatable_and_matches_cpu_oracle(self):
        torch.use_deterministic_algorithms(True,warn_only=False)
        torch.manual_seed(21)
        x=torch.randn(1,5,7,8,dtype=torch.float32,requires_grad=True)
        om=(torch.randn(1,5,7,56)*.2+.17).requires_grad_()
        args=(3,3,1,1,1,1,1,1,2,4,2.5)
        ref=grid_oracle(x,om,*args,0)
        expected=torch.autograd.grad(ref.square().mean(),(x,om))
        observations=[]
        for _ in range(2):
            a=x.detach().cuda().requires_grad_();b=om.detach().cuda().requires_grad_()
            out=dcnv4_reference(a,b,*args)
            grads=torch.autograd.grad(out.square().mean(),(a,b))
            observations.append((out.detach().cpu(),*[g.cpu() for g in grads]))
        for a,b in zip(observations[0],observations[1]):self.assertTrue(torch.equal(a,b))
        torch.testing.assert_close(observations[0][0],ref,atol=2e-6,rtol=2e-5)
        for a,b in zip(observations[0][1:],expected):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)


class EarlyStoppingTests(unittest.TestCase):
    def test_ties_do_not_extend_patience(self):
        monitor = EarlyStopping(10)
        self.assertFalse(monitor.update(.8))
        for _ in range(9):
            self.assertFalse(monitor.update(.8))
        self.assertTrue(monitor.update(.8))
        self.assertEqual(monitor.observations,11)

    def test_strict_improvement_resets_consecutive_failures(self):
        monitor = EarlyStopping(3)
        values = [.5,.4,.45,.500001,.49,.48,.47]
        self.assertEqual([monitor.update(x) for x in values], [False]*6+[True])
        self.assertEqual(monitor.best,.500001)

    def test_nonfinite_is_a_failure_not_an_early_stop_result(self):
        with self.assertRaises(ValueError):
            EarlyStopping(10).update(float('nan'))

    def test_actual_training_loop_stops_after_ten_ties_and_reloads_checkpoint(self):
        sys.path.insert(0,str(DIRECTORY.parents[1]/'tools'))
        import local_decoder_study as common
        from early_stopping import install_train
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Conv2d(3,1,1)
                torch.nn.init.zeros_(self.head.weight)
                torch.nn.init.constant_(self.head.bias,.3)
            def forward(self,x,return_aux=False):
                y = self.head(x)
                return dict(logits=y,aux_d2=y,aux_d3=y) if return_aux else y
        class Digest:
            def update(self,*args):pass
            def record(self,seed,epoch,phase):return dict(seed=seed,epoch=epoch,phase=phase)
        paired = types.SimpleNamespace(set_loader_epoch=lambda *a:None,InputDigest=Digest)
        image = torch.zeros(2,3,8,8)
        labels = torch.zeros(2,8,8);labels[:,:,::2] = 1
        batches = [(image,labels,['a','b'])]*3
        source = DIRECTORY.parents[1]/'experiments/exp00_baseline'
        factories = dict(StructureLoss_binary=torch.nn.BCEWithLogitsLoss)
        common.definitions(source/'loss.py',{'IoU_binary'},factories|dict(torch=torch,nn=torch.nn,F=F))
        namespace = dict(torch=torch,nn=torch.nn,F=F)
        common.definitions(source/'loss.py',{'IoU_binary'},namespace)
        factories['IoU_binary'] = namespace['IoU_binary']
        common.definitions(source/'kvasir_test.py',{'compute_batch_metrics'},namespace)
        factories['compute_batch_metrics'] = namespace['compute_batch_metrics']
        args = types.SimpleNamespace(epochs=100,variant='baseline',seed=1234)
        with tempfile.TemporaryDirectory() as temporary:
            records = Path(temporary)/'records';records.mkdir()
            checkpoints = Path(temporary)/'checkpoints';checkpoints.mkdir()
            with mock.patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self),\
                 mock.patch.object(common,'make_optimizer',lambda model:torch.optim.AdamW(model.parameters(),lr=.001)),\
                 mock.patch.object(common,'set_context',lambda *a:None),\
                 mock.patch.object(common,'autocast',lambda *a:contextlib.nullcontext()),\
                 mock.patch.object(common,'gradient_report',lambda model:dict(finite=all(torch.isfinite(p.grad).all() for p in model.parameters()))),\
                 mock.patch.object(common,'check_gradients',lambda report,variant:self.assertTrue(report['finite'])),\
                 contextlib.redirect_stdout(io.StringIO()):
                train,expanded = install_train(common,10)
                result = train(Net(),dict(train=batches,valid=batches),paired,factories,args,records,checkpoints)
            stop = json.loads((records/'early_stopping.json').read_text())
            self.assertEqual(stop['completed_epochs'],11)
            self.assertEqual(stop['bad_epochs'],10)
            self.assertTrue(stop['stopped_early'])
            self.assertEqual(stop['scheduler_T_max'],100)
            self.assertEqual(len((records/'epoch_metrics.jsonl').read_text().splitlines()),11)
            self.assertEqual(len(result['checkpoints']),3)
            self.assertEqual(result['best_epoch'],10) # original >= checkpoint tie rule
            self.assertAlmostEqual(result['evaluation']['iou'],33/65,places=6)


class PolyperAttentionTests(unittest.TestCase):
    def test_chunked_forward_and_gradient_match_pinned_author_code(self):
        import adapters
        adapters.setup_dependencies()
        from polyper_attention import refine_forward
        path = adapters.ASSETS/'vendor/polyper/mmsegmentation/mmseg/models/decode_heads/polyper.py'
        author = adapters.source_module('polyper_reference_test',path,
            dict(BaseDecodeHead=adapters.SegmentationHeadBase,MODELS=adapters.NoRegistry,resize=adapters.resize),
            remove_imports=('mmseg.models.decode_heads.decode_head','mmseg.registry','utils'),
            replacements=(('z_e = z_e.cuda()','z_e = z_e.to(x.device)'),
                          ('z_d = z_d.cuda()','z_d = z_d.to(x.device)')))
        torch.manual_seed(73)
        for empty in [False,True]:
            layer = author.RefineAttention(16,4,'WithBias').double()
            x = torch.randn(2,16,7,9,dtype=torch.double,requires_grad=True)
            md = (torch.rand(2,1,7,9)>.5).double()
            me = (torch.rand(2,1,7,9)>.5).double()
            if empty:me.zero_()
            reference = layer(x,md,me)
            actual = refine_forward(layer,x,md,me,query_chunk=5)
            torch.testing.assert_close(actual,reference,atol=2e-13,rtol=2e-13)
            variables = [x,layer.norm.body.weight,layer.norm.body.bias]
            ar = torch.autograd.grad(actual.square().sum(),variables,retain_graph=True)
            rr = torch.autograd.grad(reference.square().sum(),variables)
            for a,r in zip(ar,rr):torch.testing.assert_close(a,r,atol=2e-11,rtol=2e-11)


class DeterministicPoolingTests(unittest.TestCase):
    def test_global_pool_value_and_tied_gradient_match_original(self):
        from deterministic_ops import GlobalMaxPool2d
        for shape in [(2,3,4,5),(1,2,1,1)]:
            for tied in [False,True]:
                x = torch.randn(*shape,dtype=torch.double,requires_grad=True)
                if tied:x = torch.ones_like(x,requires_grad=True)
                actual = GlobalMaxPool2d()(x)
                expected = torch.nn.AdaptiveMaxPool2d(1)(x)
                self.assertTrue(torch.equal(actual,expected))
                a, = torch.autograd.grad(actual.sum(),x,retain_graph=True)
                b, = torch.autograd.grad(expected.sum(),x)
                self.assertTrue(torch.equal(a,b))


class CheckpointingTests(unittest.TestCase):
    def test_recomputation_preserves_gradients_and_single_batchnorm_update(self):
        from checkpointing import checkpoint_module
        def net():
            return torch.nn.Sequential(torch.nn.Conv2d(3,8,3,padding=1),torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),torch.nn.Dropout2d(.25),torch.nn.Conv2d(8,2,1))
        torch.manual_seed(71)
        reference,actual=net(),net()
        actual.load_state_dict(reference.state_dict())
        checkpoint_module(actual)
        x=torch.randn(2,3,7,9)
        results=[]
        for model in (reference,actual):
            torch.manual_seed(82)
            out=model(x)
            out.square().mean().backward()
            results.append((out.detach(),torch.get_rng_state().clone(),
                {k:v.clone() for k,v in model.state_dict().items()},
                {k:p.grad.clone() for k,p in model.named_parameters()}))
        for i in range(2):self.assertTrue(torch.equal(results[0][i],results[1][i]))
        for i in (2,3):
            for key,value in results[0][i].items():
                torch.testing.assert_close(value,results[1][i][key],atol=0,rtol=0)
        self.assertEqual(actual[1].num_batches_tracked.item(),1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
