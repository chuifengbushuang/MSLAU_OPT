"""Experiments 3.1/9.1 matched to the existing strict-v1 baseline.

NAF_V1_FULL_SMOKE=1 additionally checks pretrained Hiera on the server.
Only temporary synthetic fixtures/updates are used, never formal training.
"""
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_exp07_naf import definitions, load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_experiment import build, verify_base

BASELINE = 'exp00_baseline_paired_strict_v1'
P12 = 'exp03_1_naf_p12_paired_strict_v1'
DISCRETE = 'exp09_1_naf_discrete_paired_strict_v1'
CASES = (
    (P12, 'exp03_naf_p12_paired_strict_v2', 'naf_select_reg', 'PyramidNAF2D', 164),
    (DISCRETE, 'exp09_naf_discrete_paired_strict_v2', 'naf_discrete', 'CascadeDiscreteNAF2D', 56118),
)

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

if os.environ.get('NAF_V1_FULL_SMOKE') == '1' and (torch is None or A is None):
    raise ImportError('Requested server checks require torch and real Albumentations')


class DefinitionTests(unittest.TestCase):
    def test_only_input_protocol_changes_from_verified_v2_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / BASELINE
            build(BASELINE, base)
            base_config = json.loads((base / 'config.json').read_text())
            base_helper = ast.parse((base / 'paired_inputs.py').read_text())
            for experiment, previous, module, _, _ in CASES:
                source, old = (Path(tmp) / name for name in (experiment, previous))
                build(experiment, source)
                build(previous, old)
                config = json.loads((source / 'config.json').read_text())
                old_config = json.loads((old / 'config.json').read_text())
                old_config.update(id=experiment, input_protocol='paired_aug_v1')
                self.assertEqual(config, old_config)
                train = config['train_options'].copy()
                self.assertEqual(train.pop('naf_reg_weight'), .001)
                evaluation = config['eval_options'].copy()
                if experiment == P12:
                    self.assertEqual(train.pop('naf_stages'), '1,2')
                    self.assertEqual(evaluation.pop('naf_stages'), '1,2')
                self.assertEqual(train, base_config['train_options'])
                self.assertEqual(evaluation, base_config['eval_options'])
                for name in ('strict_runtime.py', 'run.py'):
                    self.assertEqual((source / name).read_bytes(), (old / name).read_bytes())
                self.assertEqual((source / 'strict_runtime.py').read_bytes(),
                                 (base / 'strict_runtime.py').read_bytes())
                self.assertEqual((source / f'networks/{module}.py').read_bytes(),
                                 (old / f'networks/{module}.py').read_bytes())
                self.assertEqual((source / 'networks/mslau_net.py').read_bytes(),
                                 (old / 'networks/mslau_net.py').read_bytes())
                helper = ast.parse((source / 'paired_inputs.py').read_text())
                def without_initial(tree):
                    return ast.dump(ast.Module(body=[node for node in tree.body
                        if not isinstance(node, ast.FunctionDef) or node.name != 'record_initial_state'],
                        type_ignores=[]))
                self.assertEqual(without_initial(helper), without_initial(base_helper))
                for name in ('get_train_transform', 'get_valid_transform', 'load_ids'):
                    nodes = [next(n for n in ast.parse((p / 'kvasir_train.py').read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == name) for p in (base, source)]
                    self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]))
                for path in source.rglob('*.py'):
                    ast.parse(path.read_text())
                launcher = load('v1_candidate_launcher', source / 'run.py')
                train_command, eval_command = launcher.commands(1234, 'preview')
                self.assertTrue(eval_command[eval_command.index('--model') + 1].endswith('/kvasir_best_model.pth'))
                self.assertIn('--naf_reg_weight', train_command)
                self.assertNotIn('input_protocol=paired_aug_v2', (source / 'kvasir_train.py').read_text())
            verify_base()


@unittest.skipIf(torch is None, 'PyTorch required')
class MatchedV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix='naf_strict_v1_')
        cls.root = Path(cls.temp.name)
        cls.sources, cls.helpers, cls.modules, cls.decoders, cls.runtimes = {}, {}, {}, {}, {}
        for experiment in (BASELINE, P12, DISCRETE):
            source = cls.root / experiment
            build(experiment, source)
            cls.sources[experiment] = source
            helper = definitions(source / 'paired_inputs.py',
                ['stable_seed', 'set_loader_epoch', 'InputDigest', 'update_tensor_digest',
                 'state_digest', 'append_record', 'record_initial_state'],
                dict(torch=torch, np=np, random=random, json=json, Path=Path, hashlib=hashlib,
                     PROTOCOL='paired_aug_v1'))
            cls.helpers[experiment] = helper
            cls.runtimes[experiment] = definitions(source / 'strict_runtime.py',
                ['configure_runtime'], dict(torch=torch, os=os))
            namespace = dict(torch=torch, nn=nn, F=F)
            if experiment != BASELINE:
                _, _, module_name, class_name, _ = next(c for c in CASES if c[0] == experiment)
                module = load('v1_' + module_name, source / f'networks/{module_name}.py')
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

    def decoder(self, experiment):
        kwargs = dict(naf_stages=(1, 2)) if experiment == P12 else {}
        return self.decoders[experiment]((4, 8, 16, 32), 4, **kwargs)

    def test_three_seeds_preserve_baseline_parameters_and_global_rng(self):
        for seed in (42, 1234, 2026):
            torch.manual_seed(seed)
            reference = self.decoder(BASELINE)
            rng = torch.get_rng_state()
            for experiment in (P12, DISCRETE):
                torch.manual_seed(seed)
                model = self.decoder(experiment)
                torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
                wrapper, original = nn.Module(), nn.Module()
                wrapper.progressive_decoder, original.progressive_decoder = model, reference
                with tempfile.TemporaryDirectory() as tmp:
                    record = self.helpers[experiment]['record_initial_state'](wrapper, seed, tmp)
                self.assertEqual(record['baseline_state_sha256'],
                                 self.helpers[BASELINE]['state_digest'](original.state_dict()))

    @unittest.skipIf(A is None, 'real Albumentations required')
    def test_v1_input_hashes_match_baseline_for_all_seeds_epochs_phases_workers(self):
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
            helpers = {e: load('v1_inputs_' + e, s / 'paired_inputs.py') for e, s in self.sources.items()}
            for seed in (42, 1234, 2026):
                for epoch in (0, 1):
                    for phase in ('train', 'valid'):
                        captured = []
                        for experiment in (BASELINE, P12, DISCRETE):
                            module = helpers[experiment]
                            name = 'get_train_transform' if phase == 'train' else 'get_valid_transform'
                            factory = definitions(self.sources[experiment] / 'kvasir_train.py',
                                [name], dict(A=A, ToTensorV2=ToTensorV2))[name]
                            data = module.PairedDataset(str(dataset), ids, factory(32), seed, phase)
                            loader = module.make_loader(data, 4, 8 if experiment == P12 else 0)
                            module.set_loader_epoch(loader, epoch)
                            digest = module.InputDigest()
                            before = torch.get_rng_state()
                            for images, masks, names in loader:
                                digest.update(images, masks, names)
                                if experiment == DISCRETE:
                                    self.modules[DISCRETE].sample_gumbels(seed, epoch, names, 1,
                                                                       'local', 60, 'cpu', torch.float32)
                            torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
                            captured.append(digest.record(seed, epoch, phase))
                        self.assertEqual(captured[0], captured[1])
                        self.assertEqual(captured[0], captured[2])
        finally:
            sys.path.remove(str(self.sources[BASELINE]))
            sys.modules.pop('loader', None)
            if previous is not None:
                sys.modules['loader'] = previous

    def test_strict_two_synthetic_decoder_updates_repeat_for_both_candidates(self):
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            self.runtimes[P12]['configure_runtime']()
        for experiment in (P12, DISCRETE):
            traces = []
            for _ in range(2):
                torch.manual_seed(1234)
                model = self.decoder(experiment).train()
                features = [torch.randn(2, c, h, h) for c, h in zip((4, 8, 16, 32), (32, 16, 8, 4))]
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0005)
                trace = []
                for step in range(2):
                    if experiment == DISCRETE:
                        model.frequency_enhancer.set_selection_context(1234, 0, ['a', 'b'])
                    optimizer.zero_grad(set_to_none=True)
                    output = model(features, 32, True)
                    loss = sum(output[k].square().mean() for k in ('logits', 'aux_d2', 'aux_d3'))
                    loss = loss + .001 * output['naf_regularization']
                    loss.backward()
                    for name, parameter in model.frequency_enhancer.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    optimizer.step()
                    trace.append((loss.item(), self.helpers[experiment]['state_digest'](model.state_dict())))
                traces.append(trace)
            self.assertEqual(traces[0], traces[1])

    @unittest.skipUnless(os.environ.get('NAF_V1_FULL_SMOKE') == '1', 'server pretrained Hiera check not requested')
    def test_full_hiera_strict_two_updates_and_eval_repeat(self):
        checkpoint = os.environ.get('HIERA_CHECKPOINT',
            '/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors')
        self.assertTrue(Path(checkpoint).is_file(), checkpoint)
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(os.environ.get('CUBLAS_WORKSPACE_CONFIG'), ':4096:8')
        self.runtimes[P12]['configure_runtime']()
        batch = int(os.environ.get('NAF_V1_SMOKE_BATCH', '12'))
        self.assertGreater(batch, 0)
        previous = {name: sys.modules.pop(name) for name in list(sys.modules)
                    if name == 'networks' or name.startswith('networks.')}
        try:
            for experiment, _, _, _, parameters in CASES:
                source = self.sources[experiment]
                traces = []
                sys.path.insert(0, str(source))
                try:
                    from networks.mslau_net import MSLAU_net
                    criterion = definitions(source / 'loss.py', ['StructureLoss_binary'],
                        dict(torch=torch, nn=nn, F=F))['StructureLoss_binary']()
                    for _ in range(2):
                        torch.manual_seed(1234)
                        kwargs = dict(naf_stages=(1, 2)) if experiment == P12 else {}
                        model = MSLAU_net(img_size=352, decoder_mode='p5_sam2unet',
                            encoder_name='sam2_hiera_large', hiera_checkpoint=checkpoint, **kwargs).cuda().train()
                        self.assertTrue(model.encoder.freeze_backbone)
                        enhancer = model.progressive_decoder.frequency_enhancer
                        self.assertEqual(sum(p.numel() for p in enhancer.parameters()), parameters)
                        groups = [[p for p in model.encoder.parameters() if p.requires_grad],
                                  [p for name, p in model.named_parameters() if not name.startswith('encoder.')]]
                        optimizer = torch.optim.AdamW([dict(params=ps, lr=.001) for ps in groups], weight_decay=.0005)
                        images = torch.randn(batch, 3, 352, 352, device='cuda')
                        target = torch.randint(0, 2, (batch, 1, 352, 352), device='cuda').float()
                        trace = []
                        for step in range(2):
                            if experiment == DISCRETE:
                                enhancer.set_selection_context(1234, 0, [f'synthetic_{i}' for i in range(batch)])
                            optimizer.zero_grad(set_to_none=True)
                            output = model(images, return_aux=True)
                            loss = sum(criterion(output[k], target) for k in ('logits', 'aux_d2', 'aux_d3'))
                            loss = loss + .001 * output['naf_regularization']
                            loss.backward()
                            self.assertTrue(torch.isfinite(loss))
                            for name, parameter in enhancer.named_parameters():
                                self.assertIsNotNone(parameter.grad, name)
                                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                            optimizer.step()
                            trace.append(dict(loss=loss.item(), state=self.helpers[experiment]['state_digest'](
                                {**{n: p for n, p in model.named_parameters() if p.requires_grad},
                                 **dict(model.named_buffers())})))
                        model.eval()
                        with torch.no_grad():
                            evaluated = model(images, return_aux=True)
                        trace.append(self.helpers[experiment]['state_digest'](
                            {k: evaluated[k] for k in ('logits', 'aux_d2', 'aux_d3')}))
                        traces.append(trace)
                        del evaluated, output, loss, optimizer, model, enhancer, groups, images, target, parameter
                        gc.collect()
                        torch.cuda.empty_cache()
                    self.assertEqual(traces[0], traces[1])
                    print('NAF_V1_FULL_SMOKE:', experiment, 'two updates and eval matched;', 'batch=', batch, flush=True)
                finally:
                    sys.path.remove(str(source))
                    for name in list(sys.modules):
                        if name == 'networks' or name.startswith('networks.'):
                            del sys.modules[name]
        finally:
            sys.modules.update(previous)


if __name__ == '__main__':
    unittest.main()
