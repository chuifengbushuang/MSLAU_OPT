import contextlib
import importlib.util
import io
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


baseline = load('baseline', ROOT / 'experiments/exp00_baseline/run.py')
organize = load('organize_legacy_results', ROOT / 'tools/organize_legacy_results.py')
restore_names = load('restore_names', ROOT / 'tools/restore_checkpoint_names.py')
builder = load('builder', ROOT / 'tools/build_experiment.py')


class WorkflowTests(unittest.TestCase):
    def test_original_source_identity(self):
        self.assertEqual(len(baseline.verify_source()['files']), 14)

    def test_original_launch_settings(self):
        train, evaluation = baseline.commands(1234, 'unit_test')
        flags = dict(zip(train[3::2], train[4::2]))
        self.assertEqual(flags['--decoder_mode'], 'p5_sam2unet')
        self.assertEqual(flags['--epoch'], '20')
        self.assertEqual(flags['--batch'], '12')
        self.assertEqual(flags['--loss'], 'structure')
        self.assertEqual(flags['--encoder_lr'], flags['--decoder_lr'])
        self.assertNotIn('--frequency_mode', train)
        self.assertNotIn('--disable_edge_guidance', train)
        self.assertIn('/data/models/le/mslau-net/checkpoints/unit_test/kvasir_best_model.pth', evaluation)

    def test_preview_does_not_launch_or_write(self):
        with patch('sys.argv', ['run.py']), patch.object(baseline.subprocess, 'Popen') as child:
            with contextlib.redirect_stdout(io.StringIO()):
                baseline.main()
            child.assert_not_called()

    def test_occupied_gpu_allows_training_and_preserves_three_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            launched = []

            def query(command, **kwargs):
                if command[0] == 'nvidia-smi':
                    return 'GPU 2: 3545 MiB used, utilization 55%, PID 120355\n'
                return 'mock runtime or environment\n'

            class Child:
                def __init__(self, command, **kwargs):
                    launched.append(command)
                    if '--output_dir' in command:
                        directory = Path(command[command.index('--output_dir') + 1])
                        for name in ['kvasir_best_model.pth', 'best_iou_0.87_epoch_19_0.3.pth',
                                     'best_loss_0.3_epoch_18_0.86.pth']:
                            (directory / name).write_bytes(name.encode())
                        self.stdout = ['Best val IoU: 0.87 at epoch 19\n']
                    else:
                        self.stdout = ['mean IoU: 0.87\n']

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def wait(self):
                    return 0

            with patch('sys.argv', ['run.py', '--gpu', '2', '--run']), \
                 patch.object(baseline, 'OUTPUT', output), \
                 patch.object(baseline, 'DATASET', tmp), patch.object(baseline, 'HIERA', __file__), \
                 patch.object(baseline.subprocess, 'check_output', side_effect=query), \
                 patch.object(baseline.subprocess, 'Popen', side_effect=Child), \
                 contextlib.redirect_stdout(io.StringIO()) as console:
                baseline.main()
            self.assertEqual(len(launched), 2)
            self.assertIn('informational only', console.getvalue())
            manifest = json.loads(next((output / 'experiments').glob('*/manifest.json')).read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(len(manifest['checkpoint_files']), 3)
            self.assertEqual(len(list((output / 'checkpoints').glob('*/*.pth'))), 3)

    def test_gpu_query_failure_is_informational(self):
        with patch.object(baseline.subprocess, 'check_output', side_effect=FileNotFoundError('nvidia-smi')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIn('unavailable', baseline.gpu_snapshot(0))

    def test_cleanup_scope_rejects_changed_or_new_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'old'
            old.mkdir()
            log = old / 'console.log'
            log.write_text('keep')
            plan = dict(source_root=tmp, remove_dirs=['old'], remove_files=[], expected_old_files={str(log): 4})
            organize.check_scope(plan)
            log.write_text('changed')
            with self.assertRaises(RuntimeError):
                organize.check_scope(plan)
            self.assertEqual(log.read_text(), 'changed')

    def test_cleanup_scope_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'old').symlink_to(root, target_is_directory=True)
            plan = dict(source_root=tmp, remove_dirs=['old'], remove_files=[], expected_old_files={})
            with self.assertRaises(RuntimeError):
                organize.check_scope(plan)

    def test_migration_failure_preserves_old_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'old/baseline_seed42'
            (source / 'checkpoints').mkdir(parents=True)
            model = source / 'checkpoints/kvasir_best_model.pth'
            model.write_bytes(b'original')
            (source / 'checkpoints/best_iou_named.pth').write_bytes(b'original')
            (source / 'checkpoints/best_loss_named.pth').write_bytes(b'bestloss')
            item = dict(id='history_baseline_seed42', source=str(source), variant='baseline', checkpoint_bytes=8, checkpoint_names=dict(fixed='kvasir_best_model.pth', best_iou='best_iou_named.pth', best_loss='best_loss_named.pth'))
            plan = dict(source_root=tmp, output_root=str(root / 'outputs'), remove_dirs=['old'], keep=[item], expected_old_files={str(p):p.stat().st_size for p in (source/'checkpoints').iterdir()})
            with patch.object(organize.subprocess, 'check_output', side_effect=RuntimeError('load failed')):
                with self.assertRaises(RuntimeError):
                    organize.migrate(plan)
            self.assertEqual(model.read_bytes(), b'original')

    def test_migration_preserves_records_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'old/baseline_seed42'
            (source / 'checkpoints').mkdir(parents=True)
            model = source / 'checkpoints/kvasir_best_model.pth'
            model.write_bytes(b'original')
            (source / 'checkpoints/best_iou_named.pth').write_bytes(b'original')
            (source / 'checkpoints/best_loss_named.pth').write_bytes(b'bestloss')
            (source / 'console.log').write_text('evidence')
            item = dict(id='history_baseline_seed42', source=str(source), variant='baseline', checkpoint_bytes=8, checkpoint_names=dict(fixed='kvasir_best_model.pth', best_iou='best_iou_named.pth', best_loss='best_loss_named.pth'))
            plan = dict(source_root=tmp, output_root=str(root / 'outputs'), remove_dirs=['old'], keep=[item], expected_old_files={str(p):p.stat().st_size for p in (source/'checkpoints').iterdir()})
            with patch.object(organize.subprocess, 'check_output', return_value='1180\n'):
                saved = organize.migrate(plan)
            self.assertEqual(Path(saved[0]['checkpoint']).read_bytes(), b'original')
            self.assertEqual(len(saved[0]['checkpoint_files']), 3)
            self.assertEqual((Path(saved[0]['checkpoint']).parent / 'best_loss_named.pth').read_bytes(), b'bestloss')
            self.assertEqual((root / 'outputs/experiments/history_baseline_seed42/console.log').read_text(), 'evidence')
            self.assertEqual(organize.migrate(plan), saved)
            self.assertTrue(model.exists())

    def test_all_nine_migrate_before_deletion_and_second_apply_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'outputs'
            output.mkdir()
            expected = {}
            keep = []
            for number in range(9):
                source = root / 'old' / f'baseline_{number}'
                (source / 'checkpoints').mkdir(parents=True)
                for name, data in [('checkpoints/kvasir_best_model.pth', b'original'),
                                   ('checkpoints/best_iou_named.pth', b'original'), ('checkpoints/best_loss_named.pth', b'bestloss'), ('console.log', b'evidence')]:
                    path = source / name
                    path.write_bytes(data)
                    expected[str(path)] = len(data)
                keep.append(dict(id=f'history_{number}', source=str(source), variant='baseline', checkpoint_bytes=8, checkpoint_names=dict(fixed='kvasir_best_model.pth', best_iou='best_iou_named.pth', best_loss='best_loss_named.pth')))
            plan = dict(source_root=tmp, output_root=str(output), remove_dirs=['old'], remove_files=[],
                        expected_old_files=expected, keep=keep, historical_summary=[{'status': 'summary'}])
            def commands(command, **kwargs):
                return '' if command[0] == 'ps' else '1180\n'
            with patch.object(organize.subprocess, 'check_output', side_effect=commands):
                with contextlib.redirect_stdout(io.StringIO()):
                    organize.apply_plan(plan)
                    organize.apply_plan(plan)
            self.assertFalse((root / 'old').exists())
            self.assertEqual(len(list((output / 'checkpoints').glob('*/kvasir_best_model.pth'))), 9)
            self.assertEqual(len(list((output / 'experiments').glob('*/console.log'))), 9)

    def test_failed_migration_never_reaches_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'old').mkdir()
            (root / 'outputs').mkdir()
            log = root / 'old/console.log'
            log.write_text('preserve')
            plan = dict(source_root=tmp, output_root=str(root / 'outputs'), remove_dirs=['old'],
                        remove_files=[], expected_old_files={str(log): 8}, keep=[])
            with patch.object(organize.subprocess, 'check_output', return_value=''), \
                 patch.object(organize, 'migrate', side_effect=RuntimeError('failed copy')):
                with self.assertRaises(RuntimeError):
                    organize.apply_plan(plan)
            self.assertEqual(log.read_text(), 'preserve')


class RestoreNamesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / 'outputs/checkpoints/history_test'
        self.directory.mkdir(parents=True)
        self.primary = self.directory / 'best_iou.pth'
        self.primary.write_bytes(b'iou state')
        self.record = self.root / 'outputs/experiments/history_test/manifest.json'
        self.record.parent.mkdir(parents=True)
        self.record.write_text(json.dumps(dict(checkpoint=str(self.primary), checkpoint_sha256=organize.digest(self.primary))))
        self.source = self.root / 'old/checkpoints'
        self.source.mkdir(parents=True)
        self.names = dict(fixed='kvasir_best_model.pth', best_iou='best_iou_named.pth', best_loss='best_loss_named.pth')
        self.plan = dict(output_root=str(self.root / 'outputs'), expected_old_files={},
                         keep=[dict(id='history_test', source=str(self.source.parent), checkpoint_names=self.names)])

    def test_preview_preserves_single_file(self):
        with contextlib.redirect_stdout(io.StringIO()):
            restore_names.restore(self.plan)
        self.assertEqual(list(self.directory.iterdir()), [self.primary])

    def test_missing_loss_is_not_fabricated_and_apply_is_repeatable(self):
        with contextlib.redirect_stdout(io.StringIO()):
            restore_names.restore(self.plan, True)
            restore_names.restore(self.plan, True)
        fixed = self.directory / self.names['fixed']
        named = self.directory / self.names['best_iou']
        self.assertEqual(fixed.read_bytes(), b'iou state')
        self.assertEqual(fixed.stat().st_ino, named.stat().st_ino)
        self.assertFalse((self.directory / self.names['best_loss']).exists())
        self.assertFalse(self.primary.exists())
        self.assertEqual(json.loads(self.record.read_text())['missing_checkpoints'], [self.names['best_loss']])

    def test_real_distinct_loss_is_recovered(self):
        loss = self.source / self.names['best_loss']
        loss.write_bytes(b'distinct loss state')
        self.plan['expected_old_files'][str(loss)] = loss.stat().st_size
        with contextlib.redirect_stdout(io.StringIO()):
            restore_names.restore(self.plan, True)
            restore_names.restore(self.plan, True)
        self.assertEqual((self.directory / loss.name).read_bytes(), b'distinct loss state')
        self.assertEqual(len(json.loads(self.record.read_text())['checkpoint_files']), 3)

    def test_conflict_stops_before_rename(self):
        (self.directory / self.names['best_iou']).write_bytes(b'wrong model')
        with self.assertRaises(RuntimeError):
            restore_names.restore(self.plan, True)
        self.assertTrue(self.primary.exists())
        self.assertFalse((self.directory / self.names['fixed']).exists())


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'repository'
        self.base = self.root / 'experiments/exp00_baseline'
        shutil.copytree(ROOT / 'experiments/exp00_baseline', self.base,
                        ignore=shutil.ignore_patterns('__pycache__'))
        for command in [['git', 'init', '-q'], ['git', 'add', '.'],
                        ['git', '-c', 'user.name=Test', '-c', 'user.email=test@local.invalid', 'commit', '-qm', 'fixture']]:
            subprocess.run(command, cwd=self.root, check=True)
        self.root_patch = patch.object(builder, 'ROOT', self.root)
        self.base_patch = patch.object(builder, 'BASE', self.base)
        self.root_patch.start()
        self.base_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.base_patch.stop)

    def define(self, name='exp01_example'):
        definition = self.root / 'experiments' / name
        definition.mkdir()
        config = json.loads((self.base / 'config.json').read_text())
        config['id'] = name
        (definition / 'config.json').write_text(json.dumps(config))
        return definition

    def test_baseline_export_runs_without_repository_or_tools(self):
        output = Path(self.temporary.name) / 'standalone'
        result = builder.build('exp00_baseline', output)
        for name, expected in json.loads((self.base / 'source_manifest.json').read_text())['files'].items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
        shutil.rmtree(self.root)
        result = subprocess.run([sys.executable, str(output / 'run.py')], cwd=output,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Preview only', result.stdout)

    def test_patch_and_added_files_are_applied_without_changing_baseline(self):
        definition = self.define()
        original = (self.base / 'kvasir_train.py').read_bytes()
        (definition / 'changes.patch').write_text(
            'diff --git a/kvasir_train.py b/kvasir_train.py\n'
            '--- a/kvasir_train.py\n+++ b/kvasir_train.py\n'
            '@@ -1,2 +1,3 @@\n+# experiment-specific change\n import argparse\n import copy\n')
        added = definition / 'added/networks/example.py'
        added.parent.mkdir(parents=True)
        added.write_text('VALUE = 1\n')
        cache = added.parent / '__pycache__'
        cache.mkdir()
        (cache / 'example.cpython-311.pyc').write_bytes(b'interpreter-specific cache')
        output = Path(self.temporary.name) / 'experiment'
        result = builder.build('exp01_example', output)
        self.assertTrue((output / 'kvasir_train.py').read_text().startswith('# experiment-specific'))
        self.assertEqual((output / 'networks/example.py').read_text(), 'VALUE = 1\n')
        self.assertEqual((self.base / 'kvasir_train.py').read_bytes(), original)
        self.assertTrue(result['patch_sha256'])
        self.assertFalse((output / 'networks/__pycache__').exists())
        self.assertEqual(set(result['added_files']), {'networks/example.py'})
        # Delete the definition too: the generated source remains self-contained.
        shutil.rmtree(definition)
        run = subprocess.run([sys.executable, str(output / 'run.py')], cwd=output,
                             text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_other_experiment_changes_do_not_affect_built_result(self):
        first = self.define()
        second = self.define('exp02_example')
        (second / 'added').mkdir()
        (second / 'added/unrelated.py').write_text('never used')
        output = Path(self.temporary.name) / 'first'
        result = builder.build('exp01_example', output)
        self.assertNotIn('unrelated.py', result['files'])
        self.assertFalse((output / 'unrelated.py').exists())

    def test_changed_baseline_is_rejected_without_output(self):
        (self.base / 'loader.py').write_text('changed')
        output = Path(self.temporary.name) / 'rejected'
        with self.assertRaises(RuntimeError):
            builder.build('exp00_baseline', output)
        self.assertFalse(output.exists())

    def test_added_file_cannot_overwrite_original(self):
        definition = self.define()
        (definition / 'added').mkdir()
        (definition / 'added/loader.py').write_text('overwritten')
        with self.assertRaises(ValueError):
            builder.build('exp01_example', Path(self.temporary.name) / 'rejected')

    def test_bad_patch_does_not_leave_output(self):
        definition = self.define()
        (definition / 'changes.patch').write_text(
            'diff --git a/loader.py b/loader.py\n--- a/loader.py\n+++ b/loader.py\n'
            '@@ -1 +1 @@\n-impossible original line\n+changed\n')
        output = Path(self.temporary.name) / 'rejected'
        with self.assertRaises(subprocess.CalledProcessError):
            builder.build('exp01_example', output)
        self.assertFalse(output.exists())

    def test_existing_export_is_never_overwritten(self):
        output = Path(self.temporary.name) / 'existing'
        output.mkdir()
        (output / 'user.txt').write_text('keep')
        with self.assertRaises(FileExistsError):
            builder.build('exp00_baseline', output)
        self.assertEqual((output / 'user.txt').read_text(), 'keep')

    def test_path_traversal_experiment_is_rejected(self):
        with self.assertRaises(ValueError):
            builder.build('../outside', Path(self.temporary.name) / 'rejected')


if __name__ == '__main__':
    unittest.main()
