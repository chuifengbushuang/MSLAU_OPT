"""Error geometry and real CPU inference with the frozen preprocessing/metrics.

The small inference fixture is not a substitute for server Hiera re-evaluation.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import analyze_baseline_errors as audit
from build_experiment import build, verify_base


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((40, 40), bool)
        self.gt[5:35, 5:35] = True

    def test_perfect_and_empty_predictions(self):
        exact = audit.diagnose(self.gt, self.gt)
        self.assertEqual(exact['error_pixels'], 0)
        self.assertEqual(exact['surface_distance_p95_px'], 0)
        self.assertEqual(exact['boundary_f1_2px'], 1)
        missing = audit.diagnose(np.zeros_like(self.gt), self.gt)
        self.assertEqual(missing['low_recall_gt_components'], 1)
        self.assertEqual(missing['empty_prediction'], 1)
        self.assertEqual(missing['boundary_f1_8px'], 0)
        self.assertIsNone(missing['surface_distance_mean_px'])
        self.assertGreater(missing['interior_fn_8px'], 0)
        empty = audit.diagnose(np.zeros_like(self.gt), np.zeros_like(self.gt))
        self.assertEqual(empty['boundary_f1_2px'], 1)
        json.dumps([exact, missing, empty], allow_nan=False)

    def test_boundary_interior_and_remote_fp_are_separate(self):
        pred = self.gt.copy()
        pred[5, 10] = False       # boundary FN
        pred[20, 20] = False     # interior FN
        pred[4, 10] = True       # adjacent FP
        pred[0, 0] = True        # distant/disjoint FP
        row = audit.diagnose(pred, self.gt)
        self.assertEqual(row['boundary_fn_2px'], 1)
        self.assertEqual(row['interior_fn_2px'], 1)
        self.assertEqual(row['boundary_fp_2px'], 1)
        self.assertEqual(row['far_fp_2px'], 1)
        self.assertEqual(row['disjoint_fp_components'], 1)
        for radius in audit.RADII:
            self.assertEqual(sum(row[f'{name}_{radius}px'] for name in
                ('boundary_fp', 'boundary_fn', 'interior_fn', 'far_fp')), row['error_pixels'])

    def test_border_objects_and_size_bin_edges(self):
        full = np.ones((5, 5), bool)
        self.assertEqual(audit.boundary(full).sum(), 16)
        self.assertEqual([audit.size_bin(x) for x in (0., .009, .01, .05, .2)],
                         ['<1%', '<1%', '1-5%', '5-20%', '>=20%'])

    def test_summary_does_not_silently_drop_perfect_or_empty_cases(self):
        rows = []
        for pred in (self.gt, np.zeros_like(self.gt)):
            rows.append(dict(audit.diagnose(pred, self.gt), iou=.5, dice=.6, precision=.7, recall=.8))
        summary = audit.summarize(rows)
        self.assertEqual(summary['images'], 2)
        self.assertEqual(summary['low_recall_images'], 1)
        for values in summary['error_partition'].values():
            self.assertAlmostEqual(sum(values['pooled_error_fraction'].values()), 1)
            self.assertAlmostEqual(sum(values['macro_error_fraction'].values()), .5)
        json.dumps(summary, allow_nan=False)


class RecordTests(unittest.TestCase):
    def test_exp09_child_keeps_experiment_selection_and_independent_output(self):
        experiment = 'exp09_naf_discrete_paired_strict_v2'
        with tempfile.TemporaryDirectory() as temporary:
            command = ['analyze_baseline_errors.py', '--experiment', experiment,
                       '--runs', 'fixture_seed1234', '--output-root', temporary]
            real_run = audit.subprocess.run
            def only_mock_python(command, **kwargs):
                if command[0] == sys.executable:
                    return None
                return real_run(command, **kwargs)
            with patch.object(sys, 'argv', command), patch.object(audit.subprocess, 'run', side_effect=only_mock_python) as run:
                audit.main()
            # build() also calls subprocess.run for git apply; inspect the final child.
            child = run.call_args.args[0]
            self.assertEqual(child[child.index('--experiment') + 1], experiment)
            self.assertTrue(child[child.index('--analysis-id') + 1].startswith('exp09_errors_v2_'))
            self.assertEqual(run.call_args.kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')

    def test_preflight_rejects_wrong_source_weights_or_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            info = build(audit.EXPERIMENT, source)
            # Minimal files exercise provenance checks without a Hiera checkpoint.
            (source / 'configs/splits/kvasir/val.txt').write_text('sample.jpg\n')
            info['files']['configs/splits/kvasir/val.txt'] = audit.sha256(source / 'configs/splits/kvasir/val.txt')
            dataset = root / 'data'
            for name in ('images', 'masks'):
                (dataset / name).mkdir(parents=True)
                (dataset / name / 'sample.jpg').write_bytes(b'fixture')
            checkpoint = root / 'checkpoints/run1/kvasir_best_model.pth'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'checkpoint fixture')
            record = root / 'experiments/run1/manifest.json'
            record.parent.mkdir(parents=True)
            metadata = dict(status='completed', source=info, input_protocol='paired_aug_v2',
                            numerical_protocol='strict_cuda_v1', seed=1234,
                            checkpoint_sha256=audit.sha256(checkpoint), evaluation_metrics={'IoU': .8, 'dice': .9})
            args = SimpleNamespace(dataset=dataset, output_root=root, runs=['run1'], experiment=audit.EXPERIMENT)
            audit.write_json(record, metadata)
            self.assertEqual(len(audit.preflight(args, source)[0]), 1)
            args.experiment = 'exp09_naf_discrete_paired_strict_v2'
            with self.assertRaisesRegex(ValueError, 'strict-v2'):
                audit.preflight(args, source)
            args.experiment = audit.EXPERIMENT
            metadata['input_protocol'] = 'paired_aug_v1'
            audit.write_json(record, metadata)
            with self.assertRaisesRegex(ValueError, 'strict-v2'):
                audit.preflight(args, source)
            metadata['input_protocol'] = 'paired_aug_v2'
            audit.write_json(record, metadata)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                audit.preflight(args, source)
            checkpoint.write_bytes(b'checkpoint fixture')
            (source / 'loader.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source differs'):
                audit.preflight(args, source)
            verify_base()

    def test_cross_seed_rejects_different_inputs(self):
        row = dict(image_id='a.jpg', size_bin='<1%', gt_area_fraction=.001, iou=.4,
                   low_recall_lt_0_5=1, far_fp_4px=3, input_sha256='same',
                   mask_sha256='same', visual_file='cases/a.jpg')
        results = audit.cross_seed({'a': [row], 'b': [dict(row, iou=.7)]})
        self.assertEqual(results[0]['runs_iou_lt_0_5'], 1)
        with self.assertRaisesRegex(ValueError, 'input mismatch'):
            audit.cross_seed({'a': [row], 'b': [dict(row, input_sha256='different')]})


class InferenceTests(unittest.TestCase):
    def test_frozen_preprocessing_metrics_and_inference_leave_parameters_unchanged(self):
        # Import only the actual evaluator functions; do not construct Hiera here.
        tree = ast.parse((ROOT / 'experiments/exp00_baseline/kvasir_test.py').read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('get_transform', 'compute_batch_metrics')]
        scope = dict(A=A, ToTensorV2=ToTensorV2, torch=torch)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'frozen evaluator', 'exec'), scope)
        spec = importlib.util.spec_from_file_location('error_fixture_loader', ROOT / 'experiments/exp00_baseline/loader.py')
        loader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader_module)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(0.))

            def forward(self, images):
                return images[:, :1] + self.bias

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('images', 'masks', 'visual/cases'):
                (root / name).mkdir(parents=True)
            ids = ['one.png', 'two.png']
            for name in ids:
                image = np.zeros((32, 32, 3), np.uint8)
                image[4:20, 4:20] = 255
                mask = np.zeros((32, 32), np.uint8)
                mask[6:22, 6:22] = 255
                cv2.imwrite(str(root / 'images' / name), image)
                cv2.imwrite(str(root / 'masks' / name), mask)
            dataset = loader_module.binary_class(str(root), ids, scope['get_transform'](32))
            loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
            model = Tiny()
            before = model.bias.detach().clone()
            rows = audit.evaluate(model, loader, SimpleNamespace(**scope), torch.device('cpu'), root / 'visual')
            self.assertEqual(len(rows), 2)
            expected_iou = (14 * 14 + 1) / (2 * 16 * 16 - 14 * 14 + 1)
            self.assertAlmostEqual(rows[0]['iou'], expected_iou, places=6)
            self.assertEqual(rows[0]['input_sha256'], rows[1]['input_sha256'])
            self.assertTrue(torch.equal(model.bias, before))
            self.assertIsNone(model.bias.grad)
            self.assertFalse(model.training)
            self.assertTrue((root / 'visual' / rows[0]['visual_file']).is_file())
            selection = audit.contact_sheets(rows, root / 'visual')
            self.assertGreater(len(selection), 0)
            self.assertFalse(list(root.rglob('*.pth')))
            json.dumps(audit.summarize(rows), allow_nan=False)


class MaskSensitivityTests(unittest.TestCase):
    def test_mask_cli_reads_jpegs_without_changing_files(self):
        import audit_mask_values as probe
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            masks = root / 'dataset/masks'
            masks.mkdir(parents=True)
            split = ROOT / 'experiments/exp00_baseline/configs/splits/kvasir/val.txt'
            ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]
            ids = [name if Path(name).suffix else name + '.jpg' for name in ids]
            raw = np.zeros((64, 64), np.uint8)
            cv2.circle(raw, (31, 31), 13, 255, -1)
            for name in ids:
                self.assertTrue(cv2.imwrite(str(masks / name), raw))
            before = [audit.sha256(masks / name) for name in ids]
            with patch.object(sys, 'argv', ['audit_mask_values.py', '--dataset', str(masks.parent),
                                           '--output-root', str(root / 'output')]):
                probe.main()
            report = next((root / 'output/experiments').glob('mask_values_v2_*/summary.json'))
            summary = json.loads(report.read_text())
            self.assertEqual(summary['images'], 120)
            self.assertTrue(summary['training_gt_unchanged'])
            self.assertGreater(summary['total_added_pixels_gt0'], 0)
            self.assertEqual([audit.sha256(masks / name) for name in ids], before)

    def test_uint8_and_float_readers_use_equivalent_thresholds(self):
        from audit_mask_values import compare_masks
        raw = np.zeros((352, 352), np.uint8)
        raw[80:120, 80:120] = 255
        raw[5, 5] = 2
        raw[20, 20] = 127
        raw[40, 40] = 128
        a, masks_a = compare_masks(raw)
        b, masks_b = compare_masks(raw.astype(np.float64) / 255)
        self.assertEqual(a, b)
        self.assertEqual(a['added_pixels_gt0'], 2)
        self.assertEqual(a['current_components'], 4)
        self.assertEqual(a['diagnostic_components'], 2)
        self.assertEqual(a['current_tiny_components_le4px'], 3)
        for x, y in zip(masks_a, masks_b):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(raw[80:120, 80:120], np.full((40, 40), 255, np.uint8))

    def test_binary_masks_are_unchanged_and_nonfinite_values_rejected(self):
        from audit_mask_values import compare_masks
        raw = np.zeros((352, 352), np.uint8)
        raw[50:100, 50:100] = 255
        values, _ = compare_masks(raw)
        self.assertEqual(values['diagnostic_mask_iou'], 1)
        self.assertEqual(values['added_pixels_gt0'], 0)
        with self.assertRaisesRegex(ValueError, 'finite grayscale'):
            compare_masks(np.full((10, 10), np.nan))


if __name__ == '__main__':
    unittest.main()
