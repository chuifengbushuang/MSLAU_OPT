"""Smoke, then unmodified experiment3 seed1234/42/2026, twenty epochs each."""
import argparse
import ast
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

import exp03_independent_study as driver
import local_decoder_study as common

OUT = common.WORKSPACE / 'analysis/20261029_exp03_independent_study'
OUTPUT = common.WORKSPACE / 'work/exp03_independent_study_20261029'
SEEDS = (1234, 42, 2026)
VARIANT = driver.VARIANTS[0]


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def locate(mode, variant, seed):
    paths = list((OUTPUT/'experiments').glob(f'local_{mode}_{variant}_seed{seed}_*'))
    if len(paths) > 1:
        raise RuntimeError(f'Multiple runs require inspection, no selection: {paths}')
    return paths[0] if paths else None


def reference_records(seed, variant):
    return driver.comparison_records(seed)[variant]


def paired_source_compatible(current, baseline):
    """Only the known, unused initial-state logging addition may differ.

    Every other AST node, including all input algorithms and constants, must
    match. The current driver records its own hashes and never calls this
    historical record_initial_state function.
    """
    allowed = ast.parse("record['baseline_state_sha256'] = state_digest({"
        "name: value for name, value in model.state_dict().items() "
        "if not name.startswith('progressive_decoder.frequency_enhancer.')})").body[0]
    def normalized(path):
        tree = ast.parse(Path(path).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'record_initial_state':
                node.body = [stmt for stmt in node.body if ast.dump(stmt) != ast.dump(allowed)]
        return ast.dump(tree)
    return normalized(current) == normalized(baseline)


def validate_references():
    checks = {}
    for seed in SEEDS:
        baseline = driver.baseline_records(seed)
        expected_config = read(baseline/'effective_config.json')
        expected_initial = read(baseline/'initial_state.json')
        for variant in ('baseline','freqselect_h','sgg_deep'):
            path = reference_records(seed,variant)
            result, config = read(path/'result.json'), read(path/'effective_config.json')
            initial = read(path/'initial_state.json')
            values = dict(completed=result['status']=='completed', epochs=len(lines(path/'epoch_metrics.jsonl'))==20,
                configuration={k:v for k,v in config.items() if k not in ('variant','naf_reg_weight')} ==
                              {k:v for k,v in expected_config.items() if k not in ('variant','naf_reg_weight')},
                regularizer=config['naf_reg_weight']==(.001 if variant=='sgg_deep' else 0),
                inputs=lines(path/'input_fingerprints.jsonl')==lines(baseline/'input_fingerprints.jsonl'),
                shared_initial=initial['shared_state_sha256']==expected_initial['shared_state_sha256'],
                cpu_rng=initial['cpu_rng_sha256']==expected_initial['cpu_rng_sha256'],
                environment=(path/'environment.txt').read_bytes()==(baseline/'environment.txt').read_bytes(),
                three_weights=len(result['checkpoints'])==3)
            for name, sha in result['checkpoints'].items():
                values['weight_'+name] = common.file_hash(path.parent.parent/'checkpoints'/path.name/name)==sha
            if not all(values.values()):
                raise RuntimeError(f'Cached reference audit failed: {seed}/{variant}: {values}')
            checks[f'{seed}/{variant}'] = dict(records=str(path),checks=values)
    return checks


def validate(path, mode, seed):
    result = read(path/('smoke.json' if mode=='smoke' else 'result.json'))
    manifest, initial = read(path/'manifest.json'),read(path/'initial_state.json')
    baseline = driver.baseline_records(seed)
    prior = read(baseline/'initial_state.json')
    config, expected = read(path/'effective_config.json'),read(baseline/'effective_config.json')
    checks = dict(completed=result['status']=='completed',
        unmodified_module=common.file_hash(path/'source/networks/naf_select_reg.py')==common.file_hash(driver.CORE)==manifest['core_sha256'],
        source_definition=manifest['source']['experiment']==driver.EXPERIMENT,
        all_snapshot_files=all(common.file_hash(path/'source'/name)==sha for name,sha in manifest['source']['files'].items()),
        driver=common.file_hash(path/'driver.py')==manifest['driver_sha256'],
        common=common.file_hash(path/'common_driver.py')==manifest['common_driver_sha256'],
        model=common.file_hash(path/'source/networks/mslau_net.py')==manifest['model_source_sha256'],
        paired_source=paired_source_compatible(path/'source/paired_inputs.py',baseline/'source/paired_inputs.py'),
        environment=(path/'environment.txt').read_bytes()==(baseline/'environment.txt').read_bytes(),
        shared_initial=initial['shared_state_sha256']==prior['shared_state_sha256'],
        cpu_rng=initial['cpu_rng_sha256']==prior['cpu_rng_sha256'],
        added_parameters=initial['trainable_parameters']==prior['trainable_parameters']+164,
        configuration={k:v for k,v in config.items() if k not in ('variant','naf_reg_weight')} ==
                      {k:v for k,v in expected.items() if k not in ('variant','naf_reg_weight')},
        regularizer=config['naf_reg_weight']==.001)
    if 'baseline_state_sha256' in prior:
        checks['all_original_initial']=initial['baseline_state_sha256']==prior['baseline_state_sha256']
    if mode=='smoke':
        smoke_audit=read(path/'original_p12_audit.json')
        checks.update(p12=smoke_audit['active_stages']==[1,2] and smoke_audit['added_parameters']==164,
            original_decoder=smoke_audit['original_decoder_logits_equal'],
            independent=smoke_audit['disabled_stages_unchanged'] and smoke_audit['no_cross_level_dependency'],
            repeated_eval=result['repeated_eval_equal'],roundtrip=result['checkpoint_roundtrip_equal'],
            full_reload=read(path/'full_reload.json')['strict_full_state_reload'])
    else:
        checks.update(epochs=len(lines(path/'epoch_metrics.jsonl'))==20,
            inputs=lines(path/'input_fingerprints.jsonl')==lines(baseline/'input_fingerprints.jsonl'),
            states=len(lines(path/'numerical_states.jsonl'))==40,
            reevaluated=abs(result['best_iou']-result['evaluation']['iou'])<1e-6,
            three_weights=len(result['checkpoints'])==3)
        for name,sha in result['checkpoints'].items():
            checks['weight_'+name]=common.file_hash(OUTPUT/'checkpoints'/path.name/name)==sha
        runtime=read(path/'initial_runtime.json')
        for variant in ('freqselect_h','sgg_deep'):
            reference=reference_records(seed,variant)
            old=read(reference/'initial_runtime.json')
            checks['initial_rng_runtime_'+variant]=runtime['runtime']==old['runtime'] and runtime['rng']==old['rng']
            checks['original_full_initial_'+variant]=initial['baseline_state_sha256']==read(reference/'initial_state.json')['baseline_state_sha256']
    if not all(checks.values()):
        raise RuntimeError(f'New run audit failed: {path}: {checks}')
    return checks


def validate_seed(seed):
    return validate(locate('train',VARIANT,seed),'train',seed)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    print('FIXED_PLAN',list(SEEDS),VARIANT,'20 epochs, cached B/H/DeepGate',flush=True)
    if not args.run:return
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    watched=[Path(driver.__file__),Path(common.__file__),driver.CORE,Path(__file__),
             common.ROOT/'tools/analyze_exp03_independent_study.py',
             common.ROOT/'experiments'/driver.EXPERIMENT/'changes.patch',
             common.ROOT/'experiments'/driver.EXPERIMENT/'config.json']
    hashes={str(p):common.file_hash(p) for p in watched}
    plan=dict(seeds=list(SEEDS),variant=VARIANT,epochs=20,unchanged_experiment=driver.EXPERIMENT,
        smoke_first=True,fresh_official_initialization=True,no_new_modules=True,no_adaptive_expansion=True,
        cached_references=['baseline','freqselect_h','sgg_deep'],hashes=hashes,output_root=str(OUTPUT))
    if (OUT/'plan.json').exists() and read(OUT/'plan.json')!=plan:
        raise RuntimeError('Existing plan differs; inspect before proceeding')
    common.write_json(OUT/'plan.json',plan)
    common.write_json(OUT/'progress.json',dict(status='reference_audit',supervisor_pid=os.getpid()))
    checks={'cached_references':validate_references()};completed=[]
    common.write_json(OUT/'audit.json',checks)
    for mode,seed in [('smoke',1234)]+[('train',s) for s in SEEDS]:
        if any(common.file_hash(p)!=hashes[str(p)] for p in watched):
            raise RuntimeError('Study implementation changed after launch')
        path=locate(mode,VARIANT,seed);log=OUT/f'{mode}_{VARIANT}_seed{seed}.console.log'
        if path is not None and not (path/('smoke.json' if mode=='smoke' else 'result.json')).is_file():
            raise RuntimeError(f'Incomplete run needs inspection; no automatic retry: {path}')
        if path is None:
            command=[sys.executable,'-u',str(Path(driver.__file__)),'--mode',mode,'--variant',VARIANT,
                     '--seed',str(seed),'--epochs','20','--batch','12','--workers','2','--threads','4',
                     '--precision','bf16','--strict','--output-root',str(OUTPUT)]
            with log.open('x') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                common.write_json(OUT/'progress.json',dict(status=mode,seed=seed,variant=VARIANT,
                    child_pid=child.pid,supervisor_pid=os.getpid(),console=str(log),completed=completed))
                print('START',mode,seed,'pid',child.pid,flush=True);code=child.wait()
            if code:raise RuntimeError(f'Worker failed with exit code {code}: {log}')
            path=locate(mode,VARIANT,seed)
        checks[f'{mode}/{seed}']=validate(path,mode,seed)
        completed.append(dict(mode=mode,seed=seed,records=str(path)))
        common.write_json(OUT/'audit.json',checks)
        print('DONE',mode,seed,json.dumps(read(path/('smoke.json' if mode=='smoke' else 'result.json'))),flush=True)
    common.write_json(OUT/'progress.json',dict(status='analysis',completed=completed))
    code=subprocess.call([sys.executable,'-u',str(common.ROOT/'tools/analyze_exp03_independent_study.py'),'--evaluate'])
    if code:raise RuntimeError('Training completed; analysis failed and requires inspection')
    common.write_json(OUT/'progress.json',dict(status='completed',completed=completed,
        summary=str(OUT/'summary.json'),report=str(common.ROOT/'studies/exp03_independent/RESULTS.md')))
    print('EXP03_INDEPENDENT_STUDY_COMPLETED',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if '--run' in sys.argv and OUT.is_dir():
            previous=read(OUT/'progress.json') if (OUT/'progress.json').exists() else {}
            common.write_json(OUT/'progress.json',dict(previous,status='supervisor_failed',error=str(error)))
        raise
