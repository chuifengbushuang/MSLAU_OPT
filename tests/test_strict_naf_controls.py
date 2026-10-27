"""Matched v2 baseline/fusion/NAF inputs, initialization and strict execution.

STRICT_NAF_FULL_SMOKE=1 enables synthetic pretrained-Hiera checks on the server.
No real-data training is launched by these tests.
"""
import ast
import contextlib
import gc
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from test_exp07_naf import definitions, load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_experiment import build, verify_base

BASELINE = 'exp00_baseline_paired_strict_v2'
FUSION = 'exp08_cascade_fusion_paired_strict_v2'
NAF = 'exp09_naf_discrete_paired_strict_v2'
CASES = ((FUSION, 'exp08_cascade_fusion', 'cascade_fusion', 'CascadeFusion2D'),
         (NAF, 'exp09_naf_discrete', 'naf_discrete', 'CascadeDiscreteNAF2D'))

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from PIL import Image
except ImportError:
    A = None

if os.environ.get('STRICT_NAF_FULL_SMOKE') == '1' and torch is None:
    raise ImportError('Requested full Hiera smoke requires PyTorch')


class DefinitionTests(unittest.TestCase):
    def test_standalone_sources_preserve_models_recipe_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build(BASELINE, root / 'baseline')
            baseline_config = json.loads((root / 'baseline/config.json').read_text())
            baseline_helper = ast.parse((root / 'baseline/paired_inputs.py').read_text())
            for experiment, original, module, _ in CASES:
                source = root / experiment
                build(experiment, source)
                config = json.loads((source / 'config.json').read_text())
                train = config['train_options'].copy()
                self.assertEqual(train.pop('naf_reg_weight'), .001 if experiment == NAF else 0.)
                self.assertEqual(train, baseline_config['train_options'])
                for key in ('base_commit', 'eval_options', 'input_protocol', 'numerical_protocol'):
                    self.assertEqual(config[key], baseline_config[key])
                self.assertEqual((source / f'networks/{module}.py').read_bytes(),
                    (ROOT / f'experiments/{original}/added/networks/{module}.py').read_bytes())
                self.assertEqual((source / 'strict_runtime.py').read_bytes(),
                                 (root / 'baseline/strict_runtime.py').read_bytes())
                helper = ast.parse((source / 'paired_inputs.py').read_text())
                def without_initial(tree):
                    return ast.dump(ast.Module(body=[n for n in tree.body
                        if not isinstance(n, ast.FunctionDef) or n.name != 'record_initial_state'], type_ignores=[]))
                self.assertEqual(without_initial(helper), without_initial(baseline_helper))
                original_tree = ast.parse((ROOT / 'experiments/exp00_baseline/networks/mslau_net.py').read_text())
                rendered_tree = ast.parse((source / 'networks/mslau_net.py').read_text())
                for name in ('SAM2UNetUp', 'RFBModified', 'BasicConv2d'):
                    nodes = [next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
                             for tree in (original_tree, rendered_tree)]
                    self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]))
                for path in source.rglob('*.py'):
                    ast.parse(path.read_text())
                launcher = load('strict_control_launcher', source / 'run.py')
                train_command, evaluation = launcher.commands(1234, 'preview')
                self.assertIn('--naf_reg_weight', train_command)
                self.assertNotIn('--naf_reg_weight', evaluation)
                self.assertTrue(evaluation[evaluation.index('--model') + 1].endswith('/kvasir_best_model.pth'))
                text = (source / 'run.py').read_text()
                reference_launcher = (root / 'baseline/run.py').read_text()
                if experiment == NAF:
                    reference_launcher = reference_launcher.replace(
                        '    metadata["input_protocol"] = config["input_protocol"]\n',
                        '    metadata["input_protocol"] = config["input_protocol"]\n'
                        '    metadata["selection_protocol"] = config["selection_protocol"]\n')
                self.assertEqual(text, reference_launcher)
                self.assertIn('CUBLAS_WORKSPACE_CONFIG=":4096:8"', text)
                self.assertIn('Incomplete or non-strict numerical audit', text)
            verify_base()


@unittest.skipIf(torch is None, 'PyTorch required')
class MatchedControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix='strict_controls_')
        cls.root = Path(cls.temp.name)
        cls.sources = {}
        cls.modules = {}
        cls.helpers = {}
        cls.runtimes = {}
        cls.decoders = {}
        for experiment in (BASELINE, FUSION, NAF):
            source = cls.root / experiment
            build(experiment, source)
            cls.sources[experiment] = source
            helper = definitions(source / 'paired_inputs.py',
                ['stable_seed', 'set_loader_epoch', 'InputDigest', 'update_tensor_digest',
                 'state_digest', 'append_record', 'record_initial_state'],
                dict(torch=torch, np=np, random=random, json=json, Path=Path, hashlib=hashlib,
                     PROTOCOL='paired_aug_v2', SEED_NAMESPACE='paired_aug_v1'))
            cls.helpers[experiment] = helper
            cls.runtimes[experiment] = definitions(source / 'strict_runtime.py',
                ['configure_runtime', 'runtime_record', 'record_setup', 'record_phase_state', 'record_selected_states'],
                dict(torch=torch, os=os, json=json, Path=Path, hashlib=hashlib, PROTOCOL='strict_cuda_v1',
                     append_record=helper['append_record'], state_digest=helper['state_digest']))
            namespace = dict(torch=torch, nn=nn, F=F)
            if experiment != BASELINE:
                _, _, module_name, class_name = next(c for c in CASES if c[0] == experiment)
                module = load('strict_' + module_name, source / f'networks/{module_name}.py')
                cls.modules[experiment] = module
                namespace[class_name] = getattr(module, class_name)
            cls.decoders[experiment] = definitions(source / 'networks/mslau_net.py',
                ['BasicConv2d', 'RFBModified', 'SAM2UNetUp', 'SAM2UNetDecoder'], namespace)['SAM2UNetDecoder']

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.flags = (torch.are_deterministic_algorithms_enabled(),
                      torch.is_deterministic_algorithms_warn_only_enabled(),
                      torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.flags[0], warn_only=self.flags[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = self.flags[2:]

    def test_three_seeds_shared_initialization_rng_and_baseline_digest(self):
        for seed in (1234, 42, 2026):
            models, rngs = {}, {}
            for experiment in (BASELINE, FUSION, NAF):
                torch.manual_seed(seed)
                models[experiment] = self.decoders[experiment]((4, 8, 16, 32), 4).eval()
                rngs[experiment] = torch.get_rng_state()
            reference = models[BASELINE].state_dict()
            for experiment in (FUSION, NAF):
                torch.testing.assert_close(rngs[experiment], rngs[BASELINE], atol=0, rtol=0)
                for name, value in reference.items():
                    torch.testing.assert_close(models[experiment].state_dict()[name], value, atol=0, rtol=0)
                wrapper = nn.Module()
                wrapper.progressive_decoder = models[experiment]
                with tempfile.TemporaryDirectory() as tmp:
                    record = self.helpers[experiment]['record_initial_state'](wrapper, seed, tmp)
                    original = nn.Module()
                    original.progressive_decoder = models[BASELINE]
                    self.assertEqual(record['baseline_state_sha256'],
                        self.helpers[BASELINE]['state_digest'](original.state_dict()))
            fusion_state = models[FUSION].frequency_enhancer.state_dict()
            naf_state = models[NAF].frequency_enhancer.state_dict()
            for name, value in fusion_state.items():
                torch.testing.assert_close(value, naf_state[name], atol=0, rtol=0)
            features = [torch.randn(2, c, h, h) for c, h in zip((4, 8, 16, 32), (32, 16, 8, 4))]
            with torch.no_grad():
                expected = models[BASELINE](features, 32, True)
                for experiment in (FUSION, NAF):
                    actual = models[experiment](features, 32, True)
                    for key in ('logits', 'aux_d2', 'aux_d3'):
                        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    @unittest.skipIf(A is None, 'real Albumentations required')
    def test_real_inputs_equal_for_three_seeds_epochs_workers_and_naf_draws(self):
        dataset = self.root / 'data'
        (dataset / 'images').mkdir(parents=True)
        (dataset / 'masks').mkdir()
        ids = [f'{i}.png' for i in range(9)]
        for i, name in enumerate(ids):
            image = np.random.default_rng(i).integers(0, 256, (37, 43, 3), dtype=np.uint8)
            mask = np.zeros((37, 43), np.uint8)
            mask[4:29, 7:32] = 255
            Image.fromarray(image).save(dataset / 'images' / name)
            Image.fromarray(mask).save(dataset / 'masks' / name)
        previous = sys.modules.pop('loader', None)
        sys.path.insert(0, str(self.sources[BASELINE]))
        try:
            for seed in (1234, 42, 2026):
                for epoch in (0, 1):
                    captured = []
                    for experiment in (BASELINE, FUSION, NAF):
                        module = load('inputs_' + experiment, self.sources[experiment] / 'paired_inputs.py')
                        factory = definitions(self.sources[experiment] / 'kvasir_train.py',
                            ['get_train_transform'], dict(A=A, ToTensorV2=ToTensorV2))['get_train_transform']
                        data = module.PairedDataset(str(dataset), ids, factory(32), seed, 'train')
                        loader = module.make_loader(data, 4, 8 if experiment == FUSION else 0)
                        module.set_loader_epoch(loader, epoch)
                        digest = module.InputDigest()
                        for images, masks, names in loader:
                            digest.update(images, masks, names)
                            if experiment == NAF:
                                self.modules[NAF].sample_gumbels(seed, epoch, names, 1, 'local', 60, 'cpu', torch.float32)
                        captured.append(digest.record(seed, epoch, 'train'))
                    self.assertEqual(captured[0], captured[1])
                    self.assertEqual(captured[0], captured[2])
        finally:
            sys.path.remove(str(self.sources[BASELINE]))
            sys.modules.pop('loader', None)
            if previous is not None:
                sys.modules['loader'] = previous

    def run_tiny_training(self, experiment, destination):
        torch.manual_seed(1234)
        destination.mkdir()
        logs = destination / 'logs'
        logs.mkdir()
        decoder = self.decoders[experiment]
        class Model(nn.Module):
            decoder_mode, conv_mla, naf_reg_weight = 'p5_sam2unet', None, .001
            def __init__(self):
                super().__init__()
                self.projections = nn.ModuleList([nn.Conv2d(3, c, 1) for c in (4, 8, 16, 32)])
                self.progressive_decoder = decoder((4, 8, 16, 32), 4)
            def forward(self, images, **kwargs):
                features = [layer(F.interpolate(images, size=(h, h))) for layer, h in
                            zip(self.projections, (32, 16, 8, 4))]
                return self.progressive_decoder(features, 32, True)
        class Dataset(torch.utils.data.Dataset):
            seed, epoch = 1234, 0
            def __init__(self, phase): self.phase = phase
            def __len__(self): return 2
            def __getitem__(self, index):
                image = torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(index+self.epoch*11))
                return image, (image[0] > 0).float(), str(index)
        model = Model()
        model.naf_reg_weight = .001 if experiment == NAF else 0.
        helper, runtime = self.helpers[experiment], self.runtimes[experiment]
        helper['record_initial_state'](model, 1234, logs)
        runtime['record_setup'](model, 1234, logs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
        loaders = {phase: torch.utils.data.DataLoader(Dataset(phase), batch_size=2,
                   generator=torch.Generator()) for phase in ('train', 'valid')}
        module = self.modules[experiment]
        ns = {**helper, **runtime, 'torch': torch, 'nn': nn, 'F': F,
              'Variable': torch.autograd.Variable, 'os': os, 'logging': logging,
              'time': time, 'FUSION_STATS_NAMES': module.FUSION_STATS_NAMES}
        if experiment == NAF:
            ns.update(BRANCHES=module.BRANCHES, FILTER_STATS_NAMES=module.FILTER_STATS_NAMES)
        definitions(self.sources[experiment] / 'kvasir_train.py',
            ['train_model', 'snapshot_state_dict', 'atomic_save_state_dict', 'log_fusion_scale_stats'], ns)
        criterion = definitions(self.sources[experiment] / 'loss.py', ['StructureLoss_binary'],
            dict(torch=torch, nn=nn, F=F))['StructureLoss_binary']()
        with contextlib.redirect_stdout(io.StringIO()):
            ns['train_model'](model, criterion, optimizer, scheduler, loaders,
                lambda x, y: (x.sigmoid() - y.unsqueeze(1)).abs().mean(), 2, str(destination),
                aux_d2_weight=1, aux_d3_weight=1, boundary_weight=0,
                input_audit_path=str(logs / 'input_fingerprints.jsonl'))
        self.assertEqual(len(list(destination.glob('*.pth'))), 3)
        for checkpoint in destination.glob('*.pth'):
            values = torch.load(checkpoint, map_location='cpu', weights_only=True)
            name = 'best_loss_sha256' if checkpoint.name.startswith('best_loss_') else 'best_iou_sha256'
            self.assertEqual(helper['state_digest'](values), json.loads((logs / 'checkpoint_tensors.json').read_text())[name])
        return {p.name: p.read_text() for p in logs.iterdir()}

    def test_actual_naf_and_fusion_training_repeats_with_strict_state_audits(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}), \
             patch('torch.cuda.is_available', return_value=False):
            self.runtimes[NAF]['configure_runtime']()
            results = {}
            for experiment in (FUSION, NAF):
                results[experiment] = [self.run_tiny_training(experiment, Path(tmp) / f'{experiment}_{i}') for i in (1, 2)]
                self.assertEqual(*results[experiment])
                states = [json.loads(s) for s in results[experiment][0]['numerical_states.jsonl'].splitlines()]
                self.assertEqual(len(states), 4)
                self.assertTrue(all(r['deterministic'] and not r['warn_only'] for r in states))
            self.assertEqual(results[FUSION][0]['input_fingerprints.jsonl'], results[NAF][0]['input_fingerprints.jsonl'])
            self.assertEqual(json.loads(results[FUSION][0]['initial_state.json'])['baseline_state_sha256'],
                             json.loads(results[NAF][0]['initial_state.json'])['baseline_state_sha256'])

    @unittest.skipUnless(os.environ.get('STRICT_NAF_FULL_SMOKE') == '1', 'server Hiera synthetic check not requested')
    def test_full_hiera_strict_two_updates_and_repeat_for_both_controls(self):
        checkpoint = os.environ.get('HIERA_CHECKPOINT',
            '/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors')
        self.assertTrue(Path(checkpoint).is_file(), checkpoint)
        self.assertEqual(os.environ.get('CUBLAS_WORKSPACE_CONFIG'), ':4096:8')
        device = torch.device(os.environ.get('STRICT_NAF_DEVICE', 'cuda'))
        batch = int(os.environ.get('STRICT_NAF_SMOKE_BATCH', '12'))
        self.assertGreater(batch, 0)
        self.runtimes[NAF]['configure_runtime']()
        previous = {name: sys.modules.pop(name) for name in list(sys.modules)
                    if name == 'networks' or name.startswith('networks.')}
        try:
            for experiment, _, _, _ in CASES:
                source = self.sources[experiment]
                traces = []
                sys.path.insert(0, str(source))
                try:
                    from networks.mslau_net import MSLAU_net
                    # Extract the actual original loss, avoiding unused torchmetrics imports.
                    criterion = definitions(source / 'loss.py', ['StructureLoss_binary'],
                        dict(torch=torch, nn=nn, F=F))['StructureLoss_binary']()
                    for repeat in range(2):
                        torch.manual_seed(1234)
                        model = MSLAU_net(img_size=352, decoder_mode='p5_sam2unet',
                            encoder_name='sam2_hiera_large', hiera_checkpoint=checkpoint).to(device).train()
                        self.assertTrue(model.encoder.freeze_backbone)
                        adapter_names = [name for name, p in model.encoder.named_parameters() if p.requires_grad]
                        self.assertTrue(adapter_names and all('.prompt_learn.' in name for name in adapter_names))
                        enhancer = model.progressive_decoder.frequency_enhancer
                        self.assertEqual(sum(p.numel() for p in enhancer.parameters()), 56118 if experiment == NAF else 55728)
                        groups = [[p for p in model.encoder.parameters() if p.requires_grad],
                                  [p for name, p in model.named_parameters() if not name.startswith('encoder.')]]
                        optimizer = torch.optim.AdamW([dict(params=ps, lr=1e-3) for ps in groups], weight_decay=5e-4)
                        owned = {id(p) for g in optimizer.param_groups for p in g['params']}
                        self.assertTrue(all(id(p) in owned for p in enhancer.parameters()))
                        image = torch.randn(batch, 3, 352, 352, device=device)
                        target = torch.randint(0, 2, (batch, 1, 352, 352), device=device).float()
                        trace = []
                        for step in range(2):
                            optimizer.zero_grad(set_to_none=True)
                            if experiment == NAF:
                                enhancer.set_selection_context(1234, 0, [f'synthetic_{i}' for i in range(batch)])
                            output = model(image, return_aux=True)
                            task = sum(criterion(output[k], target) for k in ('logits', 'aux_d2', 'aux_d3'))
                            objective = task + .001 * output['naf_regularization'] if experiment == NAF else task
                            # On step2 test task-only selector gradients after the output projection has opened.
                            (task if step else objective).backward()
                            self.assertTrue(torch.isfinite(objective))
                            for name, p in enhancer.named_parameters():
                                self.assertIsNotNone(p.grad, name)
                                self.assertTrue(torch.isfinite(p.grad).all(), name)
                            if experiment == NAF and step:
                                self.assertTrue(all(block.selector[-1].weight.grad.abs().max().item() > 0
                                                    for block in enhancer.stages.values()))
                            optimizer.step()
                            trace.append(dict(task=task.item(), objective=objective.item(),
                                logits=self.helpers[experiment]['state_digest']({k:output[k] for k in ('logits','aux_d2','aux_d3')}),
                                trainable=self.helpers[experiment]['state_digest']({n:p for n,p in model.named_parameters() if p.requires_grad}),
                                buffers=self.helpers[experiment]['state_digest'](dict(model.named_buffers()))))
                        model.eval()
                        with torch.no_grad():
                            evaluated = model(image, return_aux=True)
                        trace.append(dict(eval_logits=self.helpers[experiment]['state_digest'](
                            {k:evaluated[k] for k in ('logits','aux_d2','aux_d3')})))
                        traces.append(trace)
                        del evaluated, output, objective, task, optimizer, model, enhancer, groups, image, target, trace
                        gc.collect()
                        if device.type == 'cuda': torch.cuda.empty_cache()
                    self.assertEqual(traces[0], traces[1])
                    print('STRICT_NAF_FULL_SMOKE:', json.dumps(dict(experiment=experiment, batch=batch,
                          device=str(device), two_updates_and_eval_repeat='matched')), flush=True)
                finally:
                    sys.path.remove(str(source))
                    for name in list(sys.modules):
                        if name == 'networks' or name.startswith('networks.'):
                            del sys.modules[name]
        finally:
            sys.modules.update(previous)


if __name__ == '__main__':
    unittest.main()
