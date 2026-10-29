"""Smoke all arms, then the fixed eight authorized residual-gate runs (one original reference cached), once each."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import sgg_residual_study as driver
import local_decoder_study as common

OUT = common.WORKSPACE / 'analysis/20261029_sgg_residual_study'
OUTPUT = common.WORKSPACE / 'work/sgg_residual_study_20261029'
SEEDS = (1234, 42, 2026)
PLAN = [(seed, variant) for seed in SEEDS for variant in driver.VARIANTS
        if not (seed == 1234 and variant == 'sgg_original')]


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def locate(mode, variant, seed):
    root = OUTPUT / 'experiments'
    matches = list(root.glob(f'local_{mode}_{variant}_seed{seed}_*')) if root.is_dir() else []
    if len(matches) > 1:
        raise RuntimeError(f'Multiple runs must be inspected, never select the best: {matches}')
    return matches[0] if matches else None


def validate(path, mode, seed, variant):
    result = read(path / ('smoke.json' if mode == 'smoke' else 'result.json'))
    if result['status'] != 'completed':
        raise RuntimeError(f'Incomplete {path}')
    manifest, initial = read(path / 'manifest.json'), read(path / 'initial_state.json')
    baseline = driver.baseline_records(seed)
    reference = read(baseline / 'initial_state.json')
    checks = dict(shared_initial=initial['shared_state_sha256'] == reference['shared_state_sha256'],
        cpu_rng=initial['cpu_rng_sha256'] == reference['cpu_rng_sha256'],
        environment=(path / 'environment.txt').read_bytes() == (baseline / 'environment.txt').read_bytes(),
        snapshot_core=common.file_hash(path / 'source/networks/sgg_residual_core.py') == manifest['core_sha256'],
        snapshot_driver=common.file_hash(path / 'driver.py') == manifest['driver_sha256'],
        snapshot_common=common.file_hash(path / 'common_driver.py') == manifest['common_driver_sha256'],
        baseline_model_source=common.file_hash(path / 'source/networks/mslau_net.py') == common.file_hash(baseline / 'source/networks/mslau_net.py'),
        paired_source=common.file_hash(path / 'source/paired_inputs.py') == common.file_hash(baseline / 'source/paired_inputs.py'))
    effective, prior = read(path / 'effective_config.json'), read(baseline / 'effective_config.json')
    checks['configuration'] = {k:v for k,v in effective.items() if k not in ('variant', 'naf_reg_weight')} == {k:v for k,v in prior.items() if k not in ('variant', 'naf_reg_weight')}
    checks['selector_regularizer'] = effective['naf_reg_weight'] == .001
    checks['legacy_source'] = common.file_hash(path / 'source/networks/naf_discrete.py') == common.file_hash(common.NAF_FILE)
    if mode == 'smoke':
        checks.update(initial_logits=all(read(path / 'zero_projection_parity.json').values()),
                      full_reload=read(path / 'full_reload.json')['strict_full_state_reload'],
                      repeated_eval=result['repeated_eval_equal'],
                      roundtrip=result['checkpoint_roundtrip_equal'])
    else:
        checks.update(epochs=len(lines(path / 'epoch_metrics.jsonl')) == 20,
            inputs=lines(path / 'input_fingerprints.jsonl') == lines(baseline / 'input_fingerprints.jsonl'),
            phase_states=len(lines(path / 'numerical_states.jsonl')) == 40,
            reloaded=abs(result['evaluation']['iou'] - result['best_iou']) < 1e-6,
            three_weights=len(result['checkpoints']) == 3)
        for name, digest in result['checkpoints'].items():
            checks['weight_' + name] = common.file_hash(OUTPUT / 'checkpoints' / path.name / name) == digest
    if not all(checks.values()):
        raise RuntimeError(f'Audit failed: {path}: {checks}')
    return checks


def reference_records(seed):
    return driver.OLD_RUNS['naf_decoder'] if seed == 1234 else locate('train', 'sgg_original', seed)


def validate_cached_original():
    path = driver.OLD_RUNS['naf_decoder']
    result = read(path / 'result.json')
    baseline = driver.baseline_records(1234)
    config = read(path / 'effective_config.json')
    expected = read(baseline / 'effective_config.json')
    checks = dict(completed=result['status'] == 'completed',
        inputs=lines(path / 'input_fingerprints.jsonl') == lines(baseline / 'input_fingerprints.jsonl'),
        configuration={k:v for k,v in config.items() if k not in ('variant','naf_reg_weight')} ==
                      {k:v for k,v in expected.items() if k not in ('variant','naf_reg_weight')},
        regularizer=config['naf_reg_weight'] == .001,
        environment=(path/'environment.txt').read_bytes()==(baseline/'environment.txt').read_bytes(),
        source=common.file_hash(path/'source/networks/naf_discrete.py') == common.file_hash(common.NAF_FILE),
        epochs=len(lines(path/'epoch_metrics.jsonl')) == 20,
        three_weights=len(result['checkpoints']) == 3)
    for name, digest in result['checkpoints'].items():
        checks['weight_'+name] = common.file_hash(common.WORKSPACE/'checkpoints'/path.name/name) == digest
    if not all(checks.values()):
        raise RuntimeError(f'Cached original failed audit: {checks}')
    return dict(checks=checks, records=str(path),
        limitation='Cached seed1234 has no complete initial or 40-phase full-state hashes; no retroactive claim.')


def validate_seed(seed):
    records = {variant:locate('train',variant,seed) for variant in ('sgg_local','sgg_deep')}
    if seed != 1234:
        records['sgg_original'] = reference_records(seed)
    values = {key:read(path/'initial_state.json') for key,path in records.items()}
    if values['sgg_local'] != values['sgg_deep']:
        raise RuntimeError(f'Local/deep complete initialization differs: {seed}')
    for key in ('shared_state_sha256','baseline_state_sha256','shell_initial_sha256',
                'legacy_initial_sha256','cpu_rng_sha256'):
        if len({value[key] for value in values.values()}) != 1:
            raise RuntimeError(f'Original NAF initialization differs: {seed}/{key}')
    runtimes = [read(path/'initial_runtime.json') for path in records.values()]
    if any(v['runtime'] != runtimes[0]['runtime'] or v['rng'] != runtimes[0]['rng'] for v in runtimes[1:]):
        raise RuntimeError(f'Initial full RNG/runtime differs: {seed}')
    return dict(seed=seed, local_deep_complete_initial=True, legacy_initial=True, initial_rng_runtime=True,
        runs={v:validate(p,'train',seed,v) for v,p in records.items()},
        original_seed1234_cached=validate_cached_original() if seed==1234 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    print('FIXED_PLAN', json.dumps(PLAN), flush=True)
    if not args.run:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    watched = [Path(driver.__file__), Path(common.__file__), driver.CORE, common.NAF_FILE, Path(__file__), common.ROOT/'tools/analyze_sgg_residual_study.py']
    hashes = {str(p):common.file_hash(p) for p in watched}
    plan = dict(seeds=list(SEEDS), variants=list(driver.VARIANTS), epochs=20, runs=PLAN,
        smoke_first=True, original_decoder=True, fresh_official_initialization=True, hashes=hashes,
        no_adaptive_expansion=True, output_root=str(OUTPUT))
    plan_path = OUT / 'plan.json'
    if plan_path.exists() and read(plan_path) != json.loads(json.dumps(plan)):
        raise RuntimeError('Existing plan differs; do not silently change an experiment')
    common.write_json(plan_path, plan)
    jobs = [('smoke', 1234, v) for v in driver.VARIANTS] + [('train', s, v) for s,v in PLAN]
    completed, checks = [], {'cached_original':validate_cached_original()}
    for mode, seed, variant in jobs:
        if any(common.file_hash(p) != hashes[str(p)] for p in watched):
            raise RuntimeError('Implementation changed after the study started')
        path = locate(mode, variant, seed)
        log = OUT / f'{mode}_{variant}_seed{seed}.console.log'
        if path is not None and not (path / ('smoke.json' if mode == 'smoke' else 'result.json')).is_file():
            raise RuntimeError(f'Existing incomplete run needs inspection, no automatic retry: {path}')
        if path is None:
            command = [sys.executable, '-u', str(Path(driver.__file__)), '--mode', mode, '--variant', variant,
                '--seed', str(seed), '--epochs', '20', '--batch', '12', '--workers', '2', '--threads', '4',
                '--precision', 'bf16', '--strict', '--output-root', str(OUTPUT)]
            with log.open('x') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                common.write_json(OUT / 'progress.json', dict(status=mode, seed=seed, variant=variant,
                    child_pid=child.pid, supervisor_pid=os.getpid(), console=str(log), completed=completed))
                print('START', mode, seed, variant, 'pid', child.pid, flush=True)
                code = child.wait()
            if code:
                common.write_json(OUT / 'progress.json', dict(status='failed', mode=mode, seed=seed, variant=variant,
                    exit_code=code, completed=completed, console=str(log)))
                raise RuntimeError(f'Worker failed: {log}')
            path = locate(mode, variant, seed)
        audit = validate(path, mode, seed, variant)
        completed.append(dict(mode=mode, seed=seed, variant=variant, records=str(path)))
        checks[f'{mode}/{seed}/{variant}'] = audit
        if variant == 'sgg_deep':
            left = read(locate(mode, 'sgg_local', seed) / 'initial_state.json')
            right = read(path / 'initial_state.json')
            if left != right:
                raise RuntimeError('Local/deep complete initial state differs')
            if mode == 'train':
                checks[f'paired_seed/{seed}'] = validate_seed(seed)
            else:
                original = read(locate('smoke', 'sgg_original', seed) / 'initial_state.json')
                if any(left[key] != original[key] for key in ('legacy_initial_sha256',
                       'baseline_state_sha256', 'shell_initial_sha256', 'cpu_rng_sha256')):
                    raise RuntimeError('Smoke gates changed an original NAF initialization')
                checks['smoke_all_legacy_initial_equal'] = True
        common.write_json(OUT / 'audit.json', checks)
        result = read(path / ('smoke.json' if mode == 'smoke' else 'result.json'))
        print('DONE', mode, seed, variant, json.dumps(result.get('evaluation', {
            k:result[k] for k in ('peak_allocated_gib','estimated_20_epoch_compute_minutes') if k in result})), flush=True)
    common.write_json(OUT / 'progress.json', dict(status='training_completed', completed=completed))
    print('ALL_EIGHT_TRAININGS_COMPLETED', flush=True)
    command = [sys.executable, '-u', str(common.ROOT / 'tools/analyze_sgg_residual_study.py'), '--evaluate']
    code = subprocess.call(command)
    if code:
        common.write_json(OUT / 'progress.json', dict(status='analysis_failed', completed=completed, exit_code=code))
        raise RuntimeError('Training completed; analysis requires inspection')
    common.write_json(OUT / 'progress.json', dict(status='completed', completed=completed,
        summary=str(OUT / 'summary.json'), report=str(common.ROOT / 'studies/sgg_residual/RESULTS.md')))
    print('SGG_RESIDUAL_STUDY_COMPLETED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if '--run' in sys.argv and OUT.is_dir():
            previous = read(OUT / 'progress.json') if (OUT / 'progress.json').exists() else {}
            common.write_json(OUT / 'progress.json', dict(previous, status='supervisor_failed', error=str(error)))
        raise
