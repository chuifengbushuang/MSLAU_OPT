"""Official member lists and stopping must not alter original training updates."""
import ast
import copy
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import official_kvasir_hooks as hooks
import run_official_kvasir_study as study
import queue_official_kvasir_study as queue


class OfficialSplitTests(unittest.TestCase):
    def test_members_and_frozen_history(self):
        study.verify_base()
        manifest = json.loads((study.SPLIT / 'manifest.json').read_text())
        ids = {}
        for n in ('train','val','test'):
            p=study.SPLIT / (n+'.txt')
            self.assertEqual(study.sha(p),manifest['files'][p.name]['sha256'])
            ids[n]=p.read_text().splitlines()
        self.assertEqual((len(ids['train']),len(ids['val'])),(880,120))
        self.assertFalse(set(ids['train']) & set(ids['val']))
        self.assertEqual(ids['val'],ids['test'])
        old=(ROOT/'experiments/exp00_baseline/configs/splits/kvasir/val.txt').read_text().splitlines()
        self.assertEqual(len({Path(x).stem for x in old}&set(ids['val'])),18)

    def test_all_sources_only_add_budget_and_recording(self):
        names=(*study.ORIGINALS,'exp00_baseline_paired_strict_v2',*study.PAIRS['strict-v2'],
               *study.PAIRS['existing-v1'])
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                source=Path(d)/'source'
                study.build(name,source)
                before={p.relative_to(source).as_posix():study.sha(p) for p in source.rglob('*') if p.is_file()}
                original=(source/'kvasir_train.py').read_text()
                study.overlay(source)
                after=(source/'kvasir_train.py').read_text()
                after=after.removeprefix('from official_kvasir_hooks import configure_fp32, after_epoch, budget_loaders\nconfigure_fp32()\n')
                after=after.replace('    dataloaders = budget_loaders(dataloaders)\n','')
                after=after.replace('        if after_epoch(epoch, accuracy_list, loss_list, model, optimizer, scheduler, save_dir):\n            break\n','')
                self.assertEqual(ast.dump(ast.parse(original)),ast.dump(ast.parse(after)))
                for p,h in before.items():
                    if p not in ('kvasir_train.py','kvasir_test.py'):
                        self.assertEqual(study.sha(source/p),h,p)

    def test_early_stopping_ties_reset_and_nonfinite(self):
        stop=hooks.EarlyStopping(10)
        self.assertFalse(stop.update(.7))
        for _ in range(9):self.assertFalse(stop.update(.7))
        self.assertFalse(stop.update(.71))
        self.assertEqual(stop.bad_epochs,0)
        for _ in range(9):self.assertFalse(stop.update(.70))
        self.assertTrue(stop.update(.70))
        with self.assertRaises(ValueError):stop.update(float('nan'))

    def test_formal_loader_identity_and_smoke_limit(self):
        loaders={'train':[1,2,3,4],'valid':[5,6,7]}
        with mock.patch.dict(os.environ,OFFICIAL_KVASIR_SMOKE_STEPS='0'):
            self.assertIs(hooks.budget_loaders(loaders),loaders)
        with mock.patch.dict(os.environ,OFFICIAL_KVASIR_SMOKE_STEPS='2'):
            limited=hooks.budget_loaders(loaders)
            self.assertEqual(list(limited['train']),[1,2])
            self.assertEqual(len(limited['valid']),2)

    def test_queue_requires_success_and_process_release(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'progress.json';path.write_text('{"status":"completed"}')
            dep=dict(name='prior',state_file=str(path),expected={'status':'completed'},
                     processes=[dict(pid=12,start_ticks='345')])
            with mock.patch.object(queue,'same_process',return_value=True):
                self.assertFalse(queue.dependency_state(dep)['ready'])
            with mock.patch.object(queue,'same_process',return_value=False):
                self.assertTrue(queue.dependency_state(dep)['ready'])
            path.write_text('{"status":"training"}')
            with mock.patch.object(queue,'same_process',return_value=False):
                with self.assertRaises(RuntimeError):queue.dependency_state(dep)
            path.write_text('{')
            self.assertFalse(queue.dependency_state(dep)['ready'])

    def test_queue_rejects_failed_prior_task_and_changed_sources(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'failure.json';path.write_text('{}')
            with self.assertRaises(RuntimeError):
                queue.dependency_state(dict(name='prior',failure_file=str(path)))
            with self.assertRaises(RuntimeError):
                queue.verify_signature({'source_sha256':{str(path):'incorrect'}})

    def test_original_cpu_updates_and_three_weights_preserved(self):
        import torch
        from torch import nn
        torch.set_num_threads(1)
        text=(ROOT/'experiments/exp00_baseline/kvasir_train.py').read_text()
        nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in
               ('snapshot_state_dict','atomic_save_state_dict','log_fusion_scale_stats','train_model')]
        class Tiny(nn.Module):
            decoder_mode='p5_sam2unet'
            def __init__(self):
                super().__init__();self.layer=nn.Conv2d(3,1,1);self.conv_mla=nn.Identity()
            def forward(self,x,**kw):
                y=self.layer(x)
                return dict(logits=y,aux_d2=y,aux_d3=y)
        def run(directory,patched):
            import time
            env=dict(torch=torch,nn=nn,copy=copy,os=os,time=time,logging=logging,
                     Variable=torch.autograd.Variable,after_epoch=hooks.after_epoch)
            tree=ast.Module(body=copy.deepcopy(nodes),type_ignores=[])
            if patched:
                for n in tree.body:
                    if n.name=='train_model':
                        loop=next(x for x in n.body if isinstance(x,ast.For))
                        extra=ast.parse('if after_epoch(epoch, accuracy_list, loss_list, model, optimizer, scheduler, save_dir):\n    break').body[0]
                        loop.body.append(extra)
            ast.fix_missing_locations(tree)
            exec(compile(tree,'original_train_cpu_fixture','exec'),env)
            torch.manual_seed(71);model=Tiny()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0005)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
            batch=(torch.ones(2,3,4,4),torch.zeros(2,4,4),['a','b'])
            loaders={'train':[batch],'valid':[batch]}
            metric=lambda x,y:torch.tensor(.5)
            directory.mkdir()
            records=directory/'records';records.mkdir()
            hooks._stopper=None
            with mock.patch.dict(os.environ,OFFICIAL_KVASIR_RECORDS=str(records),
                    OFFICIAL_KVASIR_PATIENCE='10',OFFICIAL_KVASIR_MAX_EPOCHS='100'),\
                 mock.patch.object(torch.cuda,'is_available',return_value=False),\
                 mock.patch.object(torch.cuda,'max_memory_allocated',return_value=0),\
                 mock.patch.object(torch.cuda,'max_memory_reserved',return_value=0):
                _,losses,scores=env['train_model'](model,nn.BCEWithLogitsLoss(),optimizer,
                    scheduler,loaders,metric,100 if patched else 11,str(directory))
            return model.state_dict(),losses,scores,list(directory.glob('*.pth')),scheduler.state_dict()
        with tempfile.TemporaryDirectory() as d:
            a=run(Path(d)/'original',False);b=run(Path(d)/'patched',True)
            self.assertEqual(len(b[1]['train']),11)
            self.assertEqual((a[1],a[2],a[4]),(b[1],b[2],b[4]))
            for k in a[0]:self.assertTrue(torch.equal(a[0][k],b[0][k]),k)
            self.assertEqual((len(a[3]),len(b[3])),(3,3))


if __name__=='__main__':unittest.main()
