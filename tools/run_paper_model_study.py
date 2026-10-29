"""Test every architecture, then run the fixed, serial early-stopping study."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import paper_model_study as driver
import analyze_paper_models


def signature():
    files = [*(p for p in driver.DEFINITION.iterdir() if p.is_file()),
             Path(driver.__file__),Path(__file__),Path(analyze_paper_models.__file__),
             driver.ROOT/'tools/prepare_paper_models.py',
             driver.ROOT/'tests/test_paper_models.py']
    return {str(p.relative_to(driver.ROOT)):driver.common.file_hash(p) for p in files}


def validate(path,mode,model,seed):
    if (path/'failure.json').exists():raise RuntimeError(f'Failed run: {path}')
    config=driver.read(path/'effective_config.json')
    if config['model']!=model or config['seed']!=seed or config['epochs']!=100 or config['early_stopping']['patience']!=10:
        raise RuntimeError(f'Wrong run protocol: {path}')
    if mode=='smoke':
        result=driver.read(path/'smoke.json')
        reload=driver.read(path/'full_reload.json')
        if not (result['status']=='completed' and result['batch']==12 and len(result['steps'])==6
                and result['repeated_eval_equal'] and result['checkpoint_roundtrip_equal'] and reload['strict_complete_state_reload']):
            raise RuntimeError(f'Smoke gate failed: {path}')
    else:
        result=driver.read(path/'result.json');stop=driver.read(path/'early_stopping.json')
        epochs=[json.loads(x) for x in (path/'epoch_metrics.jsonl').read_text().splitlines()]
        values=[e['valid']['iou'] for e in epochs]
        from early_stopping import EarlyStopping
        monitor=EarlyStopping(10)
        decisions=[monitor.update(v) for v in values]
        if any(decisions[:-1]) or (len(epochs)<100 and not decisions[-1]) or len(epochs)>100:
            raise RuntimeError('Training did not follow patience=10 / maximum100')
        if len(epochs)!=stop['completed_epochs'] or abs(max(values)-result['evaluation']['iou'])>1e-6:
            raise RuntimeError('Best checkpoint/epoch count mismatch')
        if len(result['checkpoints'])!=3:raise RuntimeError('Missing original three checkpoint files')
        for name,digest in result['checkpoints'].items():
            if driver.common.file_hash(driver.OUTPUT/'checkpoints'/path.name/name)!=digest:
                raise RuntimeError(f'Checkpoint content changed: {name}')
        if len((path/'input_fingerprints.jsonl').read_text().splitlines())!=2*len(epochs):
            raise RuntimeError('Missing complete input trajectory')
        if len((path/'numerical_states.jsonl').read_text().splitlines())!=2*len(epochs):
            raise RuntimeError('Missing phase-state observations')
    return True


def locate(mode,model,seed,current=None):
    candidates=list((driver.OUTPUT/'experiments').glob(f'paper_{mode}_{model}_seed{seed}_*'))
    if mode=='train' and len(candidates)>1:raise RuntimeError('Duplicate formal run; inspect instead of selecting scores')
    valid=[]
    for path in candidates:
        if (path/'failure.json').exists():
            if mode=='train':raise RuntimeError(f'Formal run failed; inspect before retry: {path}')
            continue
        marker=path/('smoke.json' if mode=='smoke' else 'result.json')
        if not marker.exists():
            if mode=='train':raise RuntimeError(f'Existing incomplete formal run: {path}')
            continue
        if mode=='smoke' and current is not None:
            manifest=driver.read(path/'manifest.json')
            expected={p.name:driver.common.file_hash(p) for p in driver.DEFINITION.iterdir() if p.is_file()}
            if manifest['definition_sha256']!=expected:continue
            if driver.common.file_hash(path/'driver.py')!=driver.common.file_hash(driver.__file__):continue
        validate(path,mode,model,seed);valid.append(path)
    return sorted(valid)[-1] if valid else None


def execute(mode,model,seed):
    logfile=driver.OUT/f'{mode}_{model}_seed{seed}.log'
    command=[sys.executable,str(Path(driver.__file__)), '--mode',mode,'--model',model,'--seed',str(seed)]
    with logfile.open('a') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=driver.WORKSPACE)
        driver.common.write_json(driver.OUT/'progress.json',dict(status=mode,model=model,seed=seed,
            supervisor_pid=os.getpid(),child_pid=child.pid,log=str(logfile)))
        print('START',mode,model,seed,child.pid,flush=True)
        code=child.wait()
    if code:raise RuntimeError(f'{mode}/{model}/{seed} failed: {logfile}')
    path=locate(mode,model,seed)
    if path is None:raise RuntimeError('Child returned without complete output')
    validate(path,mode,model,seed)
    print('DONE',mode,model,seed,str(path),flush=True)
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args(argv)
    driver.OUT.mkdir(parents=True,exist_ok=True)
    if not args.run:
        print(json.dumps(dict(models=driver.VARIANTS,seeds=driver.SEEDS,maximum_epochs=100,patience=10,
            order=[(m,s) for m in driver.VARIANTS for s in driver.SEEDS],formal_training_started=False)))
        return
    sys.path.insert(0,str(driver.DEFINITION))
    with (driver.OUT/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=signature()
        driver.common.write_json(driver.OUT/'plan.json',dict(models=driver.VARIANTS,seeds=driver.SEEDS,
            cap=100,patience=10,source_signature=current,supervisor_pid=os.getpid()))
        try:
            with (driver.OUT/'assets_verification.log').open('w') as log:
                subprocess.run([sys.executable,str(driver.ROOT/'tools/prepare_paper_models.py')],
                    cwd=driver.WORKSPACE,stdout=log,stderr=subprocess.STDOUT,check=True)
            for test in ['test_paper_models.py','test_workflow.py']:
                with (driver.OUT/test.replace('.py','.log')).open('w') as log:
                    environment=dict(os.environ,PAPER_MODELS_CUDA_TESTS='1')
                    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',test,'-v'],
                        cwd=driver.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,env=environment)
            smokes={}
            # Hard gate: no formal model starts until all five GPU smokes pass.
            for model in driver.VARIANTS:
                path=locate('smoke',model,1234,current) or execute('smoke',model,1234)
                smokes[model]=str(path)
            driver.common.write_json(driver.OUT/'gate.json',dict(status='passed',source_signature=current,
                smokes=smokes,tests=['test_paper_models.py','test_workflow.py']))
            for model in driver.VARIANTS:
                for seed in driver.SEEDS:
                    if signature()!=current:raise RuntimeError('Code changed after preflight; stop before training')
                    path=locate('train',model,seed) or execute('train',model,seed)
                    validate(path,'train',model,seed)
                    analyze_paper_models.analyze()
            result=analyze_paper_models.analyze()
            if result['status']!='completed':raise RuntimeError('Missing formal runs')
            driver.common.write_json(driver.OUT/'progress.json',result)
            print('COMPLETED',json.dumps(result),flush=True)
        except BaseException as error:
            driver.common.write_json(driver.OUT/'supervisor_failure.json',dict(error=str(error),pid=os.getpid()))
            raise


if __name__=='__main__':main()
