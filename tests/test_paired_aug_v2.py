"""Actual Albumentations RNG coupling regression and shared-stream input checks."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_experiment import build, verify_base
from check_baseline_numerics import definitions

try:
    import albumentations as A
    import numpy as np
    import torch
    from albumentations.pytorch import ToTensorV2
    from PIL import Image
except ImportError:
    A = None


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DefinitionTests(unittest.TestCase):
    def test_execution_and_training_recipe_unchanged(self):
        old, new = [ROOT / 'experiments' / name for name in
                    ('exp00_baseline_paired_strict_v1', 'exp00_baseline_paired_strict_v2')]
        for name in ('changes.patch', 'added/strict_runtime.py'):
            self.assertEqual((old / name).read_bytes(), (new / name).read_bytes())
        a, b = [json.loads((p / 'config.json').read_text()) for p in (old, new)]
        b['id'], b['input_protocol'] = a['id'], a['input_protocol']
        self.assertEqual(a, b)
        self.assertEqual(len(verify_base()['files']), 14)


@unittest.skipIf(A is None, 'real Albumentations/torch/skimage required')
class AugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.sources = [cls.root / name for name in ('v1', 'v2')]
        for source, experiment in zip(cls.sources, ('exp00_baseline_paired_strict_v1', 'exp00_baseline_paired_strict_v2')):
            build(experiment, source)
        sys.path.insert(0, str(cls.sources[1]))
        cls.modules = [load('rng_' + source.name, source / 'paired_inputs.py') for source in cls.sources]
        cls.factories = definitions(cls.sources[1] / 'kvasir_train.py',
            {'get_train_transform', 'get_valid_transform'}, dict(A=A, ToTensorV2=ToTensorV2))
        cls.ids = [f'{i}.png' for i in range(13)]
        cls.dataset = cls.root / 'dataset'
        for folder in ('images', 'masks'):
            (cls.dataset / folder).mkdir(parents=True)
        for i, name in enumerate(cls.ids):
            image = np.random.default_rng(i).integers(0, 256, (37, 43, 3), dtype=np.uint8)
            mask = np.zeros((37, 43), dtype=np.uint8)
            mask[4:29, 7:32] = 255
            Image.fromarray(image).save(cls.dataset / 'images' / name)
            Image.fromarray(mask).save(cls.dataset / 'masks' / name)
        # Only this temporary fixture export receives miniature split lists.
        for phase, ids in (('train', cls.ids), ('val', cls.ids[:4])):
            (cls.sources[1] / f'configs/splits/kvasir/{phase}.txt').write_text('\n'.join(ids) + '\n')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.sources[1]))
        sys.modules.pop('loader', None)
        cls.temp.cleanup()

    def dataset_for(self, version=1, phase='train'):
        factory = self.factories['get_train_transform' if phase == 'train' else 'get_valid_transform']
        return self.modules[version].PairedDataset(str(self.dataset), self.ids, factory(32), 1234, phase)

    def capture(self, workers, epoch):
        module = self.modules[1]
        loader = module.make_loader(self.dataset_for(), 4, workers)
        module.set_loader_epoch(loader, epoch)
        digest = module.InputDigest()
        for images, masks, ids in loader:
            digest.update(images, masks, ids)
        return digest.record(1234, epoch, 'train')

    def test_current_coupling_is_reproduced_and_shared_stream_restores_combinations(self):
        image = np.ones((25, 31, 3), dtype=np.float32) * 127
        mask = np.zeros((25, 31), dtype=np.float32)
        summaries = []
        for version in (0, 1):
            data = self.dataset_for(version)
            gates = []
            for i in range(2048):
                seed = self.modules[version].stable_seed(1234, 'augmentation', 'train', i // 880, str(i % 880) + '.jpg')
                if version == 0:
                    data.transforms.set_random_seed(seed)
                else:
                    self.modules[version].seed_compose(data.transforms, seed)
                data.transforms(image=image, mask=mask)
                gates.append(tuple(bool(data.transforms.transforms[j].params) for j in (1, 2, 3, 4)))
            summaries.append(np.asarray(gates))
        old, new = summaries
        self.assertTrue(np.array_equal(old[:, 0], old[:, 1]))
        self.assertTrue(np.array_equal(old[:, 0], old[:, 2]))
        self.assertEqual(len({tuple(row) for row in old}), 3)
        self.assertEqual(len({tuple(row) for row in new}), 16)
        for marginal in new.mean(0)[:3]:
            self.assertTrue(0.20 < marginal < 0.30)
        self.assertTrue(0.45 < new.mean(0)[3] < 0.55)
        self.assertTrue(0.04 < (new[:, 0] & new[:, 1]).mean() < 0.085)
        self.assertTrue(0.007 < new[:, :3].all(1).mean() < 0.03)

    def test_sampler_worker_and_sample_root_seeds_are_preserved(self):
        old, new = self.modules
        for epoch in range(20):
            for stream, parts in [('sample_order', (epoch,)), ('worker_base', ('train', epoch)),
                                  ('augmentation', ('train', epoch, self.ids[0]))]:
                self.assertEqual(old.stable_seed(1234, stream, *parts), new.stable_seed(1234, stream, *parts))
            a, b = [module.make_loader(self.dataset_for(i), 4, 0) for i, module in enumerate(self.modules)]
            old.set_loader_epoch(a, epoch)
            new.set_loader_epoch(b, epoch)
            self.assertEqual(list(a.sampler), list(b.sampler))
            torch.testing.assert_close(a.generator.get_state(), b.generator.get_state(), atol=0, rtol=0)

    def test_workers_global_rng_access_order_and_epoch(self):
        a = self.capture(0, 0)
        torch.rand(431); np.random.rand(97); random.random()
        self.assertEqual(a, self.capture(8, 0))
        self.assertEqual(a, self.capture(0, 0))
        self.assertNotEqual(a['images_sha256'], self.capture(0, 1)['images_sha256'])
        data = self.dataset_for()
        first = data[0]
        data[7]
        again = data[0]
        for x, y in zip(first[:2], again[:2]):
            torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_mask_alignment_validation_and_global_rng_preservation(self):
        transform = A.Compose([A.HorizontalFlip(p=0.25), A.VerticalFlip(p=0.25), A.RandomRotate90(p=0.25)])
        mask = np.zeros((25, 25), np.uint8); mask[3:11, 7:20] = 1
        image = np.repeat(mask[..., None], 3, axis=2)
        for seed in (1, 1234, (1 << 63) - 1):
            transform.set_random_state(np.random.default_rng(seed), random.Random(seed))
            out = transform(image=image, mask=mask)
            np.testing.assert_array_equal(out['image'][..., 0], out['mask'])
        old, new = (self.dataset_for(i, 'valid') for i in (0, 1))
        for index in range(4):
            a, b = old[index], new[index]
            for x, y in zip(a[:2], b[:2]):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
        before = torch.get_rng_state()
        self.dataset_for()[3]
        torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)

    def test_independent_processes_reproduce_actual_input_capture(self):
        captures = []
        for i in range(2):
            output = self.root / f'capture_{i}.json'
            subprocess.run([sys.executable, str(self.sources[1] / 'check_inputs.py'), '--capture',
                '--dataset', str(self.dataset), '--seed', '1234', '--workers', '0', '--epochs', '2',
                '--perturb', str(i * 13), '--output', str(output)], cwd=self.sources[1], check=True,
                stdout=subprocess.DEVNULL,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', NO_ALBUMENTATIONS_UPDATE='1'))
            captures.append(json.loads(output.read_text()))
        self.assertEqual(captures[0]['records'], captures[1]['records'])
        self.assertEqual(captures[0]['augmentation_probes'], captures[1]['augmentation_probes'])

    def test_full_cpu_audit_includes_coupling_regression_and_rejects_bad_statistics(self):
        output = self.root / 'complete_input_audit.json'
        subprocess.run([sys.executable, str(self.sources[1] / 'check_inputs.py'),
            '--dataset', str(self.dataset), '--seed', '1234', '--workers', '8', '--epochs', '2',
            '--output', str(output)], cwd=self.sources[1], check=True, stdout=subprocess.DEVNULL,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', NO_ALBUMENTATIONS_UPDATE='1'))
        result = json.loads(output.read_text())
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['input_protocol'], 'paired_aug_v2')
        self.assertEqual(result['captures'][0]['augmentation_decisions']['combinations'], 16)
        result['captures'][0]['augmentation_decisions']['horizontal_and_vertical'] = 0.25
        validate = definitions(self.sources[1] / 'check_inputs.py', {'validate'}, {})['validate']
        with self.assertRaisesRegex(RuntimeError, 'decisions remain coupled'):
            validate(result['captures'])


if __name__ == '__main__':
    unittest.main()
