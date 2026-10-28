"""Execute only the user-approved, conditional seeds42/2026 confirmation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import analyze_fredft_study as analysis
import fredft_study as driver
import local_decoder_study as common


def completed(variant, seed):
    matches = [p for p in (driver.WORKSPACE / 'experiments').glob(f'local_train_{variant}_seed{seed}_*')
               if (p / 'result.json').is_file()]
    if len(matches) > 1:
        raise RuntimeError(f'Multiple completed runs; do not choose a repeat: {matches}')
    return matches[0] if matches else None


def validate_seed(seed, records):
    baseline = records['baseline']
    initial = analysis.read(baseline / 'initial_state.json')
    inputs = analysis.lines(baseline / 'input_fingerprints.jsonl')
    config = analysis.read(baseline / 'effective_config.json')
    for variant, path in records.items():
        result = analysis.read(path / 'result.json')
        other = analysis.read(path / 'initial_state.json')
        effective = analysis.read(path / 'effective_config.json')
        assert result['status'] == 'completed' and result['seed'] == seed
        assert len(analysis.lines(path / 'epoch_metrics.jsonl')) == 20
        assert len(inputs) == 40 and analysis.lines(path / 'input_fingerprints.jsonl') == inputs
        for key in ('shared_state_sha256', 'baseline_state_sha256', 'cpu_rng_sha256'):
            assert other[key] == initial[key], (seed, variant, key)
        assert {k:v for k,v in effective.items() if k != 'variant'} == {k:v for k,v in config.items() if k != 'variant'}
        assert abs(result['best_iou'] - result['evaluation']['iou']) < 1e-6
        assert len(result['checkpoints']) == 3
    assert analysis.read(records['fredft_frequency'] / 'initial_state.json') == analysis.read(records['fredft_spatial'] / 'initial_state.json')
    return dict(seed=seed, all40_inputs=True, baseline_initial=True, fs_complete_initial=True, reevaluation=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    summary = analysis.summarize(analysis.find_runs())
    if not summary['expand_multiseed']:
        raise SystemExit('Conditional confirmation not triggered: F must exceed baseline and S.')
    first = analysis.find_runs()
    for variant in ('fredft_frequency', 'fredft_spatial'):
        manifest = analysis.read(first[variant] / 'manifest.json')
        if manifest['core_sha256'] != common.file_hash(driver.CORE) or manifest['driver_sha256'] != common.file_hash(Path(driver.__file__)):
            raise RuntimeError('Candidate implementation changed after seed1234')
    plan = [(seed, variant) for seed in (42, 2026) for variant in ('baseline', 'fredft_frequency', 'fredft_spatial')]
    print('CONFIRMATION_PLAN', json.dumps(plan), flush=True)
    if not args.run:
        return
    status = analysis.OUT / 'confirmation_progress.json'
    finished, seed_checks = [], []
    for seed in (42, 2026):
        records = {}
        for variant in ('baseline', 'fredft_frequency', 'fredft_spatial'):
            path = completed(variant, seed)
            if path is None:
                log = driver.WORKSPACE / 'work' / f'fredft_confirmation_{variant}_seed{seed}.console.log'
                command = [sys.executable, '-u', str(Path(driver.__file__)), '--mode', 'train', '--variant', variant,
                           '--seed', str(seed), '--epochs', '20', '--batch', '12', '--workers', '2',
                           '--threads', '4', '--precision', 'bf16']
                common.write_json(status, dict(status='training', seed=seed, variant=variant,
                    completed=finished, console=str(log), supervisor_pid=os.getpid()))
                print('START', seed, variant, str(log), flush=True)
                with log.open('w') as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    common.write_json(status, dict(status='training', seed=seed, variant=variant,
                        completed=finished, console=str(log), supervisor_pid=os.getpid(), training_pid=child.pid))
                    code = child.wait()
                if code:
                    common.write_json(status, dict(status='failed', seed=seed, variant=variant, exit_code=code,
                        completed=finished, console=str(log)))
                    raise RuntimeError(f'Training failed: {seed}/{variant}; see {log}')
                path = completed(variant, seed)
                if path is None:
                    raise RuntimeError('Training returned without a completed result')
            records[variant] = path
            finished.append(dict(seed=seed, variant=variant, records=str(path)))
            print('DONE', seed, variant, json.dumps(analysis.read(path / 'result.json')['evaluation']), flush=True)
        seed_checks.append(validate_seed(seed, records))
    common.write_json(status, dict(status='completed', completed=finished, seed_checks=seed_checks))
    print('CONFIRMATION_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
