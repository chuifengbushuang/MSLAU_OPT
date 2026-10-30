"""Optional run arguments preserve frozen sources and original training updates."""
import ast
import copy
import importlib.util
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from build_experiment import build,verify_base
from experiment_overrides import apply_overrides
import queue_official_kvasir_study as queue
SPLIT=ROOT/'configs/splits/kvasir_medico_official_880_120_v1'


class OfficialSplitTests(unittest.TestCase):
    def test_original_default_is_unchanged(self):
        verify_base()
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';build('exp00_baseline',source)
            before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
            apply_overrides(source)
            for p,data in before.items():self.assertEqual(p.read_bytes(),data)

    def test_original_path_arguments_and_all_experiment_overrides(self):
        for config in sorted((ROOT/'experiments').glob('*/config.json')):
            with self.subTest(config=config.parent.name),tempfile.TemporaryDirectory() as d:
                source=Path(d)/'source';build(config.parent.name,source)
                originals={p:p.read_bytes() for p in source.rglob('*.py') if p.name not in ('run.py','kvasir_train.py','kvasir_test.py')}
                apply_overrides(source,split_dir=SPLIT,epochs=100,patience=10,fp32=True)
                for p,data in originals.items():self.assertEqual(p.read_bytes(),data)
                for p in source.rglob('*.py'):ast.parse(p.read_text())
                spec=importlib.util.spec_from_file_location('preview_launcher',source/'run.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                module.verify_source()
                train,evaluation=module.commands(1234,'test')
                self.assertEqual(train[train.index('--train_txt')+1],str(SPLIT/'train.txt'))
                self.assertEqual(train[train.index('--val_txt')+1],str(SPLIT/'val.txt'))
                self.assertEqual(evaluation[evaluation.index('--split_txt')+1],str(SPLIT/'val.txt'))
                self.assertEqual(train[train.index('--epoch')+1],'100')
                if 'fingerprints' in (source/'run.py').read_text():
                    self.assertIn("max(r['epoch'] for r in fingerprints)",(source/'run.py').read_text())
        verify_base()

    def test_official_members(self):
        ids={n:(SPLIT/(n+'.txt')).read_text().splitlines() for n in ('train','val','test')}
        self.assertEqual((len(set(ids['train'])),len(set(ids['val']))),(880,120))
        self.assertFalse(set(ids['train'])&set(ids['val']))
        self.assertEqual(ids['val'],ids['test'])

    def test_queue_waits_for_success_and_process_exit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'progress.json';p.write_text('{"status":"completed"}')
            dep=dict(name='prior',state_file=str(p),expected={'status':'completed'},processes=[dict(pid=12,start_ticks='3')])
            with mock.patch.object(queue,'same_process',return_value=True):self.assertFalse(queue.dependency_state(dep)['ready'])
            with mock.patch.object(queue,'same_process',return_value=False):self.assertTrue(queue.dependency_state(dep)['ready'])
            p.write_text('{"status":"training"}')
            with mock.patch.object(queue,'same_process',return_value=False):
                with self.assertRaises(RuntimeError):queue.dependency_state(dep)

    def test_cpu_early_stop_matches_original_eleven_epochs_and_three_weights(self):
        import torch,time
        from torch import nn
        torch.set_num_threads(1)
        class Tiny(nn.Module):
            decoder_mode='p5_sam2unet'
            def __init__(self):
                super().__init__();self.layer=nn.Conv2d(3,1,1);self.conv_mla=nn.Identity()
            def forward(self,x,**kw):
                y=self.layer(x);return dict(logits=y,aux_d2=y,aux_d3=y)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';build('exp00_baseline',source)
            original=(source/'kvasir_train.py').read_text()
            apply_overrides(source,epochs=100,patience=10)
            patched=(source/'kvasir_train.py').read_text()
            def run(text,folder,epochs):
                nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in
                    ('snapshot_state_dict','atomic_save_state_dict','log_fusion_scale_stats','train_model')]
                env=dict(torch=torch,nn=nn,copy=copy,os=os,time=time,logging=logging,Variable=torch.autograd.Variable)
                exec(compile(ast.Module(body=nodes,type_ignores=[]),'cpu_fixture','exec'),env)
                torch.manual_seed(71);model=Tiny();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0005)
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
                batch=(torch.ones(2,3,4,4),torch.zeros(2,4,4),['a','b']);folder.mkdir()
                with mock.patch.object(torch.cuda,'is_available',return_value=False):
                    _,losses,scores=env['train_model'](model,nn.BCEWithLogitsLoss(),optimizer,scheduler,
                        {'train':[batch],'valid':[batch]},lambda x,y:torch.tensor(.5),epochs,str(folder))
                return model.state_dict(),losses,scores,list(folder.glob('*.pth')),scheduler.state_dict()
            a=run(original,root/'original',11);b=run(patched,root/'patched',100)
            self.assertEqual(len(b[1]['train']),11)
            self.assertEqual((a[1],a[2],a[4]),(b[1],b[2],b[4]))
            for k in a[0]:self.assertTrue(torch.equal(a[0][k],b[0][k]),k)
            self.assertEqual((len(a[3]),len(b[3])),(3,3))


if __name__=='__main__':unittest.main()
