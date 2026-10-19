"""Output routing and failure handling without importing model dependencies."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import inventory_outputs, run_experiment


def test_dry_run_has_no_side_effects(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'outputs'
    monkeypatch.setattr(sys, 'argv', ['run', '--name', 'baseline_seed42_r1',
        '--gpu', '0', '--root', str(root), '--dry-run', '--', '--seed', '42'])
    assert run_experiment.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert not root.exists()
    assert all(Path(path).name == plan['run_id'] for path in plan['paths'].values())
    assert plan['command'][-4:] == ['--output_dir', plan['paths']['checkpoints'],
                                   '--log_dir', plan['paths']['experiments'] + '/logs']


def test_managed_path_cannot_escape_layout(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run', '--name', 'bad', '--gpu', '0',
                                     '--dry-run', '--', '--output_dir=/elsewhere'])
    with pytest.raises(SystemExit):
        run_experiment.main()


def test_evaluation_routes_images_and_links_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['run', '--name', 'eval_seed42', '--gpu', '0',
        '--kind', 'evaluate', '--parent-run', 'training-id', '--root', str(tmp_path),
        '--dry-run', '--', '--model', '/models/best.pth', '--debug', 'true'])
    run_experiment.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan['parent_run'] == 'training-id'
    assert plan['command'][-2:] == ['--debug_dir', plan['paths']['visualization']]
    assert '--output_dir' not in plan['command']


def test_busy_gpu_fails_before_training_and_keeps_record(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'kvasir_train.py').write_text('raise AssertionError("must not run")')
    def capture(command, cwd):
        if command[:2] == ['git', 'ls-files']:
            return 'kvasir_train.py\n'
        if '--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu' in command:
            return '0, GPU-test, RTX 3090, 580.178.04, 4096, 50\n'
        return ''
    monkeypatch.setattr(run_experiment, 'capture', capture)
    root = tmp_path / 'outputs'
    monkeypatch.setattr(sys, 'argv', ['run', '--name', 'busy', '--gpu', '0',
        '--repo', str(repo), '--root', str(root)])
    with pytest.raises(RuntimeError, match='busy'):
        run_experiment.main()
    manifest = next((root / 'experiments').glob('*/manifest.json'))
    assert json.loads(manifest.read_text())['status'] == 'failed'
    assert not (manifest.parent / 'console.log').exists()


@pytest.mark.parametrize('exitcode', [0, 7])
def test_child_completion_and_failure_are_recorded(tmp_path, monkeypatch, exitcode):
    repo = tmp_path / 'repo'
    repo.mkdir()
    entry = repo / 'kvasir_train.py'
    entry.write_text('import argparse,sys\nfrom pathlib import Path\n'
        'p=argparse.ArgumentParser(); p.add_argument("--output_dir"); p.add_argument("--log_dir"); a=p.parse_args()\n'
        'Path(a.output_dir,"kvasir_best_model.pth").write_bytes(b"test checkpoint")\n'
        'print("Best val IoU: 0.875000 at epoch 17")\n'
        f'sys.exit({exitcode})\n')
    def capture(command, cwd):
        if command[:2] == ['git', 'ls-files']:
            return 'kvasir_train.py\n'
        if command[:2] == ['git', 'rev-parse']:
            return 'test-commit\n'
        return ''
    monkeypatch.setattr(run_experiment, 'capture', capture)
    original_run = subprocess.run
    def runtime(command, **kwargs):
        if len(command) > 1 and command[1] == '-c':
            return subprocess.CompletedProcess(command, 0, '{"cuda_available": false}\n')
        return original_run(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', runtime)
    root = tmp_path / 'outputs'
    monkeypatch.setattr(sys, 'argv', ['run', '--name', 'stub_seed42_r1', '--gpu', 'cpu',
        '--repo', str(repo), '--root', str(root)])
    assert run_experiment.main() == exitcode
    manifest = next((root / 'experiments').glob('*/manifest.json'))
    record = json.loads(manifest.read_text())
    assert record['status'] == ('completed' if exitcode == 0 else 'failed')
    assert record['returncode'] == exitcode
    assert record['artifacts'][0]['sha256']
    assert (manifest.parent / 'source.zip').is_file()
    assert record['best_validation_iou'] == [['0.875000', '17']]
    assert len((root / 'experiments/index.jsonl').read_text().splitlines()) == 1


def test_inventory_does_not_mutate_inputs_and_excludes_its_report(tmp_path, monkeypatch):
    root = tmp_path / 'old'
    run = root / 'smoke_run'
    run.mkdir(parents=True)
    weight = run / 'best.pth'
    weight.write_bytes(b'preserve')
    (run / 'console.log').write_text('failed runs are evidence too')
    output = root / 'audit'
    monkeypatch.setattr(sys, 'argv', ['inventory', '--root', str(root), '--output-dir', str(output)])
    inventory_outputs.main()
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['files'] == 2
    assert summary['observed_weight_files'] == 1
    assert weight.read_bytes() == b'preserve'
    assert 'REVIEW_CLEANUP' in (output / 'files.csv').read_text()
