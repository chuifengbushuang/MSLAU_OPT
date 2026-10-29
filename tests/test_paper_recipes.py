"""Native recipes: upstream numeric oracles, stage transfer, DP4 and data isolation."""
import copy
import json
import math
import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
import random
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'studies/paper_recipes'))
sys.path.insert(0,str(ROOT/'tools'))
import recipes as r
import data as d
import paper_recipe_study as driver


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Module()
        for i in range(4):
            setattr(self.network,f'layer{i}',nn.Sequential(nn.Conv2d(3,3,1),nn.BatchNorm2d(3)))
        self.head=nn.Conv2d(3,1,1)


class TinyASG(nn.Module):
    def __init__(self):
        super().__init__()
        class Network(nn.Module):
            def __init__(self):
                super().__init__();self.conv=nn.Conv2d(3,4,1);self.bn=nn.BatchNorm2d(4)
                self.heads=nn.ModuleList([nn.Conv2d(4,1,1) for _ in range(9)])
            def forward(self,x):
                x=self.bn(self.conv(x)).relu()
                return tuple(head(x) for head in self.heads)
        self.network=Network()


class RecipesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_paper_numeric_parameters_and_complete_groups(self):
        for name,stage,lrs,cls,wd in [('polyper',2,[.0002,.0002],torch.optim.AdamW,.0001),
            ('eat',2,[.001,.001],torch.optim.AdamW,.05),('condseg',1,[.0001],torch.optim.Adam,0),
            ('condseg',2,[.00001,.0001],torch.optim.Adam,0),('asgnet',2,[.0001],torch.optim.Adam,0)]:
            model=Tiny();opt=r.optimizer(model,name,stage)
            self.assertIsInstance(opt,cls);self.assertEqual([g['lr'] for g in opt.param_groups],lrs)
            actual=[id(p) for g in opt.param_groups for p in g['params']]
            self.assertEqual(set(actual),{id(p) for p in model.parameters()});self.assertEqual(len(actual),len(set(actual)))
            self.assertEqual(max(g['weight_decay'] for g in opt.param_groups),wd)
        model=Tiny();opt=r.optimizer(model,'eat')
        self.assertEqual(opt.defaults['betas'],(.9,.95))

    def test_polyper_scheduler_against_mmengine_oracle(self):
        r.adapters.setup_dependencies()
        from mmengine.optim import LinearLR,PolyLR
        oracle=torch.optim.AdamW([nn.Parameter(torch.ones(1))],lr=.0002)
        warm=LinearLR(oracle,start_factor=1e-6,by_epoch=False,begin=0,end=1500)
        poly=PolyLR(oracle,eta_min=0,power=1,begin=1500,end=160000,by_epoch=False)
        ours=torch.optim.AdamW([nn.Parameter(torch.ones(1))],lr=.0002);schedule=r.Schedule(ours,'polyper')
        for step in range(1601):
            schedule.before_step()
            self.assertAlmostEqual(ours.param_groups[0]['lr'],oracle.param_groups[0]['lr'],places=13,msg=str(step))
            schedule.after_step();warm.step();poly.step()

    def test_asg_paper_lr_milestone(self):
        opt=r.optimizer(Tiny(),'asgnet');schedule=r.Schedule(opt,'asgnet')
        for epoch,expected in [(0,1e-4),(78,1e-4),(79,1e-5),(99,1e-5)]:
            schedule.before_epoch(epoch);self.assertAlmostEqual(opt.param_groups[0]['lr'],expected)

    def test_eat_scheduler_matches_author_after_epoch(self):
        opt=r.optimizer(Tiny(),'eat');schedule=r.Schedule(opt,'eat')
        self.assertEqual(opt.param_groups[0]['lr'],0.)
        schedule.after_epoch(0,1.);self.assertEqual(opt.param_groups[0]['lr'],0.)
        schedule.after_epoch(1,1.);self.assertEqual(opt.param_groups[0]['lr'],.001)
        schedule.after_epoch(100,1.)
        self.assertAlmostEqual(opt.param_groups[0]['lr'],.001/2*(1+math.cos(math.pi*98/3998)))

    def test_eat_monai_native_gradients(self):
        criterion=r.NativeLoss('eat')
        x=torch.randn(2,1,8,8,requires_grad=True);y=torch.rand_like(x)
        loss=criterion(x,y);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(x.grad.abs().max().item(),0)
        self.assertEqual(criterion.dice.smooth_nr,0);self.assertEqual(criterion.dice.smooth_dr,1e-5)

    def test_condseg_scheduler_logging_is_json_safe_without_mutating_training(self):
        model=Tiny();opt=r.optimizer(model,'condseg',2);schedule=r.Schedule(opt,'condseg')
        reference_opt=r.optimizer(copy.deepcopy(model),'condseg',2)
        reference=torch.optim.lr_scheduler.ReduceLROnPlateau(reference_opt,mode='min',patience=30)
        for epoch in range(40):
            before=copy.deepcopy(schedule.native.state_dict())
            snapshot=schedule.state_dict()
            json.dumps(snapshot,allow_nan=False)
            self.assertEqual(schedule.native.state_dict(),before)
            self.assertEqual(snapshot['native']['mode_worse'],'Infinity')
            if epoch==0:
                self.assertEqual(snapshot['native']['best'],'Infinity')
            schedule.after_epoch(epoch,1.)
            reference.step(1.)
            self.assertEqual([g['lr'] for g in opt.param_groups],[g['lr'] for g in reference_opt.param_groups])
            self.assertEqual(schedule.native.state_dict(),reference.state_dict())

    def test_condseg_consistency_against_author_class(self):
        namespace=r.definitions(r.ASSETS/'vendor/condseg/utils/run_engine_stage1.py',{'BinaryConsistencyLoss'},dict(torch=torch,nn=nn,F=F))
        criterion=r.NativeLoss('condseg',1)
        a=torch.rand(4,1,8,8,requires_grad=True);b=torch.rand_like(a,requires_grad=True);target=torch.rand_like(a)
        actual=criterion(a,target,augmented=b)
        expected=criterion.dice_bce(a,target)+criterion.dice_bce(b,target)+namespace['BinaryConsistencyLoss']()(a,b)
        self.assertTrue(torch.equal(actual,expected));actual.backward()
        self.assertGreater(a.grad.abs().max().item(),0);self.assertGreater(b.grad.abs().max().item(),0)

    def test_condseg_stage2_against_executed_author_train(self):
        class OutputModel(nn.Module):
            def __init__(self):
                super().__init__();self.values=nn.Parameter(torch.randn(4,4,1,8,8))
            def forward(self,x):
                return tuple(self.values.sigmoid().unbind(0))
        model=OutputModel();other=copy.deepcopy(model)
        criterion=r.NativeLoss('condseg',2)
        x=torch.rand(4,3,8,8);fg=torch.rand(4,1,8,8);bg=1-fg
        namespace=r.definitions(r.ASSETS/'vendor/condseg/utils/run_engine.py',{'complementary_loss','train'},
            dict(torch=torch,nn=nn,F=F,np=np,tqdm=lambda value,**kw:value,calculate_metrics=lambda *a:[0]*4))
        opt=torch.optim.SGD(other.parameters(),lr=0.)
        expected=namespace['train'](other,[(x,(fg,bg))],opt,criterion.dice_bce,'cpu')[0]
        actual=criterion(model(x),fg,background=bg);actual.backward()
        self.assertEqual(expected,actual.item());torch.testing.assert_close(model.values.grad,other.values.grad,rtol=1e-6,atol=1e-7)
        self.assertTrue((model.values.grad.flatten(1).abs().amax(1)>0).all())

    def test_condseg_stage_transfer_includes_bn(self):
        source,destination=Tiny(),Tiny()
        source.network.layer0[1].running_mean.fill_(3.)
        head=destination.head.weight.detach().clone()
        result=r.transfer_encoder(source,destination)
        self.assertGreater(result['tensors'],0)
        self.assertTrue(torch.equal(destination.network.layer0[1].running_mean,source.network.layer0[1].running_mean))
        self.assertTrue(torch.equal(head,destination.head.weight))

    def test_asg_output_selection_is_final_head(self):
        values=tuple(torch.full((1,1,2,2),float(i)) for i in range(9))
        self.assertIs(r.primary(values,'asgnet'),values[4])

    def test_asg_all_nine_heads_receive_native_loss_gradients(self):
        values=tuple(torch.randn(2,1,8,8,requires_grad=True) for _ in range(9))
        loss=r.NativeLoss('asgnet')(values,torch.rand(2,1,8,8),edge=torch.rand(2,1,8,8),fraction=.5)
        loss.backward()
        for item in values:
            self.assertGreater(item.grad.abs().max().item(),0)

    def test_virtual_dp4_gradient_bn_and_adam_step_equivalence(self):
        torch.manual_seed(90)
        model=TinyASG().double();oracle=copy.deepcopy(model)
        replicas=[copy.deepcopy(model) for _ in range(4)]
        x=torch.randn(40,3,8,8,dtype=torch.double);target=torch.rand(40,1,8,8,dtype=torch.double);edge=torch.rand_like(target)
        loss=r.NativeLoss('asgnet');a=torch.optim.Adam(model.parameters(),lr=1e-4);b=torch.optim.Adam(oracle.parameters(),lr=1e-4)
        r.backward_asgnet(model,x,target,edge,loss,.5)
        for i,replica in enumerate(replicas):
            sl=slice(i*10,(i+1)*10)
            (loss(replica.network(x[sl]),target[sl],edge=edge[sl],fraction=.5)/4).backward()
        for j,p in enumerate(oracle.parameters()):
            p.grad=sum(list(replica.parameters())[j].grad for replica in replicas)
        for actual,expected in zip(model.parameters(),oracle.parameters()):
            torch.testing.assert_close(actual.grad,expected.grad,rtol=1e-12,atol=1e-12)
        for actual,expected in zip(model.buffers(),replicas[0].buffers()):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        a.step();b.step()
        for actual,expected in zip(model.parameters(),oracle.parameters()):
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(model.network.bn.num_batches_tracked.item(),1)

    def test_whole_stage_checkpoint_two_views_preserves_bn_and_gradients(self):
        from checkpointing import checkpoint_module
        torch.manual_seed(77)
        direct=TinyASG().double();checked=copy.deepcopy(direct)
        checkpoint_module(checked.network)
        x=torch.randn(2,3,8,8,dtype=torch.double)
        for model in (direct,checked):
            first=model.network(x);second=model.network(x.flip(-1))
            loss=sum(v.square().mean() for v in first)+sum(v.abs().mean() for v in second)
            loss.backward()
        for actual,expected in zip(checked.parameters(),direct.parameters()):
            torch.testing.assert_close(actual.grad,expected.grad,rtol=0,atol=0)
        for actual,expected in zip(checked.buffers(),direct.buffers()):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(checked.network.bn.num_batches_tracked.item(),2)

    def test_real_stage1_projection_commutation_matches_author_double(self):
        direct=r.make_model('condseg',1,pretrained=False,checkpointing=False,efficient_stage1=False).double()
        checked=copy.deepcopy(direct)
        import types
        checked.network.forward=types.MethodType(r.efficient_stage1_forward,checked.network)
        x=torch.randn(2,3,32,32,dtype=torch.double)
        a=direct.network(x);b=checked.network(x)
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
        target=torch.rand_like(a)
        F.binary_cross_entropy(a,target).backward();F.binary_cross_entropy(b,target).backward()
        self.assertEqual(set(direct.state_dict()),set(checked.state_dict()))
        for (name,actual),(_,expected) in zip(checked.named_parameters(),direct.named_parameters()):
            torch.testing.assert_close(actual.grad,expected.grad,rtol=1e-8,atol=1e-9,msg=name)
        for actual,expected in zip(checked.buffers(),direct.buffers()):
            torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)

    def test_original_shared_split_and_full_polyper_batches(self):
        ids=d.split_ids();self.assertEqual(len(ids['train']),880)
        sampler=d.ContinuousEpochSampler(1234)
        first=list(sampler);second=list(sampler)
        self.assertEqual(len(first),882);self.assertEqual(len({i for i,_ in first[:880]}),880)
        self.assertNotEqual(first,second)

    def test_eval_masks_exact_original_reader_resize(self):
        import albumentations as A
        for name in r.POLICY:
            dataset=d.NativeDataset(name,False,1234)
            for i in (0,31,119):
                item=dataset[i]
                from skimage import io
                mask=(io.imread(d.DATA/'masks'/item['id'],as_gray=True)>0).astype(np.float32)
                expected=A.Resize(352,352)(image=np.zeros((*mask.shape,3),np.uint8),mask=mask)['mask']
                np.testing.assert_array_equal(item['mask'][0].numpy(),expected)

    def test_native_inputs_reproducible_and_352(self):
        for name in r.POLICY:
            for stage in ((1,2) if name=='condseg' else (2,)):
                dataset=d.NativeDataset(name,True,42,stage)
                a=dataset[0];random.seed(999);np.random.seed(999);b=dataset[0]
                for key in ('image','mask','background','edge'):
                    self.assertTrue(torch.equal(a[key],b[key]),(name,stage,key))
                    self.assertTrue(torch.isfinite(a[key]).all())
                self.assertEqual(a['image'].shape,(3,352,352))
                dataset.epoch=1;c=dataset[0]
                self.assertFalse(torch.equal(a['image'],c['image']),(name,stage))

    def test_asg_edge_generation_against_author_algorithm(self):
        mask=np.zeros((40,40),np.uint8);mask[8:32,8:32]=255
        body=cv2.distanceTransform(mask,cv2.DIST_L2,5)**.5;body[body>2]=255
        expected=cv2.blur(mask-body,(3,3))
        # OpenCV write/read performs its native saturating round-to-uint8 cast.
        ok,encoded=cv2.imencode('.png',expected)
        self.assertTrue(ok)
        np.testing.assert_array_equal(d.edge_from_author(mask),cv2.imdecode(encoded,0))

    def test_metrics_match_original_function(self):
        official=r.definitions(ROOT/'experiments/exp00_baseline/kvasir_test.py',{'compute_batch_metrics'},dict(torch=torch))['compute_batch_metrics']
        probability=torch.rand(3,1,12,12);mask=(torch.rand_like(probability)>.5).float()
        rows=driver.metric_rows(probability,mask,['a','b','c']);expected=official((probability>.5).float(),mask)
        for key in ('iou','dice','accuracy','precision','recall','f1'):
            self.assertEqual([row[key] for row in rows],expected[key])

    def test_queue_aggregation_partial_complete_and_mismatched_metadata(self):
        import run_paper_recipe_study as queue
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'analysis';output=Path(temporary)/'runs';out.mkdir()
            scope={'baseline_reference':[dict(seed=seed,path=f'baseline{seed}',evaluation=dict(iou=.87,dice=.92))
                for seed in driver.SEEDS],'fixed':{'maximum_epochs':100},'deferred':{'l2s':'wait for official code'}}
            driver.write(out/'scope.json',scope)
            def completed(model,seed):
                path=output/'experiments'/f'native_train_{model}_seed{seed}_fixture'
                driver.write(path/'result.json',dict(model=model,seed=seed,status='completed',evaluation=dict(iou=.7,dice=.78)))
                return path
            with patch.object(driver,'OUT',out),patch.object(driver,'OUTPUT',output):
                first=completed('polyper',1234)
                partial=queue.aggregate()
                self.assertEqual(partial['status'],'running')
                self.assertEqual(len(partial['runs']),4)
                self.assertEqual(partial['runs'][-1]['model'],'polyper')
                self.assertEqual(partial['runs'][-1]['seed'],1234)
                for model in driver.MODELS:
                    for seed in driver.SEEDS:
                        completed(model,seed)
                full=queue.aggregate()
                self.assertEqual(full['status'],'completed')
                self.assertEqual(len(full['runs']),15)
                self.assertTrue(all(row['seeds']==3 for row in full['summary']))
                wrong=driver.read(first/'result.json');wrong['seed']=42
                driver.write(first/'result.json',wrong)
                with self.assertRaisesRegex(ValueError,'metadata differs'):
                    queue.aggregate()


if __name__=='__main__':
    unittest.main()
