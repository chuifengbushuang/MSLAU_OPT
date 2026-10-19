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
    spec.loader.exec_module(module)
    return module


baseline = load('baseline', ROOT / 'experiments/exp00_baseline/run.py')
organize = load('organize', ROOT / 'tools/organize_legacy_results.py')
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
            item = dict(id='history_baseline_seed42', source=str(source), variant='baseline', checkpoint_bytes=8)
            plan = dict(source_root=tmp, output_root=str(root / 'outputs'), remove_dirs=['old'], keep=[item])
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
            (source / 'console.log').write_text('evidence')
            item = dict(id='history_baseline_seed42', source=str(source), variant='baseline', checkpoint_bytes=8)
            plan = dict(source_root=tmp, output_root=str(root / 'outputs'), remove_dirs=['old'], keep=[item])
            with patch.object(organize.subprocess, 'check_output', return_value='1180\n'):
                saved = organize.migrate(plan)
            self.assertEqual(Path(saved[0]['checkpoint']).read_bytes(), b'original')
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
                                   ('checkpoints/best_loss.pth', b'discard'), ('console.log', b'evidence')]:
                    path = source / name
                    path.write_bytes(data)
                    expected[str(path)] = len(data)
                keep.append(dict(id=f'history_{number}', source=str(source), variant='baseline', checkpoint_bytes=8))
            plan = dict(source_root=tmp, output_root=str(output), remove_dirs=['old'], remove_files=[],
                        expected_old_files=expected, keep=keep, historical_summary=[{'status': 'summary'}])
            def commands(command, **kwargs):
                return '' if command[0] == 'ps' else '1180\n'
            with patch.object(organize.subprocess, 'check_output', side_effect=commands):
                with contextlib.redirect_stdout(io.StringIO()):
                    organize.apply_plan(plan)
                    organize.apply_plan(plan)
            self.assertFalse((root / 'old').exists())
            self.assertEqual(len(list((output / 'checkpoints').glob('*/best_iou.pth'))), 9)
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
        output = Path(self.temporary.name) / 'experiment'
        result = builder.build('exp01_example', output)
        self.assertTrue((output / 'kvasir_train.py').read_text().startswith('# experiment-specific'))
        self.assertEqual((output / 'networks/example.py').read_text(), 'VALUE = 1\n')
        self.assertEqual((self.base / 'kvasir_train.py').read_bytes(), original)
        self.assertTrue(result['patch_sha256'])
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
