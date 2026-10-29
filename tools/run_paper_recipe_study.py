"""Gate every recipe first, then execute the twelve authorized formal runs."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

import paper_recipe_study as study


def find(mode,model,seed):
    paths=sorted((study.OUTPUT/'experiments').glob(f'native_{mode}_{model}_seed{seed}_*'))
    if mode=='train' and len(paths)>1:
        raise RuntimeError('Duplicate formal runs require inspection')
    for path in reversed(paths):
        if (path/'failure.json').exists():
            if mode=='train':
                raise RuntimeError(f'Failed formal run must be inspected: {path}')
            continue
        marker=path/('smoke.json' if mode=='smoke' else 'result.json')
        if marker.exists():
            if mode=='smoke':
                current=study.signature()
                if study.read(marker)['signature']!=current or study.read(path/'config.json')['source_signature']!=current:
                    continue
            return path
        if mode=='train':
            raise RuntimeError(f'Incomplete formal run must be inspected: {path}')
    return None


def execute(mode,model,seed):
    logfile=study.OUT/f'{mode}_{model}_seed{seed}.log'
    with logfile.open('a') as stream:
        child=subprocess.Popen([sys.executable,str(Path(study.__file__)),'--mode',mode,'--model',model,'--seed',str(seed)],
            cwd=study.WORKSPACE,stdout=stream,stderr=subprocess.STDOUT)
        study.write(study.OUT/'progress.json',dict(status=mode,model=model,seed=seed,child_pid=child.pid,
            supervisor_pid=os.getpid(),log=str(logfile)))
        print('START',mode,model,seed,child.pid,flush=True)
        code=child.wait()
    if code:
        raise RuntimeError(f'{mode}/{model}/{seed} failed: {logfile}')
    result=find(mode,model,seed)
    if result is None:
        raise RuntimeError('No completed output from child')
    print('DONE',mode,model,seed,str(result),flush=True)
    return result


def aggregate():
    import csv
    import numpy as np
    scope=study.read(study.OUT/'scope.json')
    completed=[]
    for model in study.MODELS:
        for seed in study.SEEDS:
            path=find('train',model,seed)
            if path:
                result=study.read(path/'result.json')
                if result.get('model')!=model or result.get('seed')!=seed or result.get('status')!='completed':
                    raise ValueError(f'Completed run metadata differs from its queue slot: {path}')
                completed.append(dict(result,records=str(path)))
    rows=[dict(model='baseline',seed=item['seed'],records=item['path'],evaluation=item['evaluation'])
          for item in scope['baseline_reference']]+completed
    reported={'baseline':(None,None),'polyper':(.9036,.9482),'eat':(.9785,.9709),
              'condseg':(.846,.905),'asgnet':(.870,.921)}
    summary=[]
    for model in ('baseline',*study.MODELS):
        selected=[r for r in rows if r['model']==model]
        if not selected:
            continue
        iou,dice=reported[model]
        summary.append(dict(model=model,seeds=len(selected),
            iou_mean_percent=float(np.mean([r['evaluation']['iou'] for r in selected])*100),
            iou_sd_percent=float(np.std([r['evaluation']['iou'] for r in selected],ddof=1)*100) if len(selected)>1 else None,
            dice_mean_percent=float(np.mean([r['evaluation']['dice'] for r in selected])*100),
            dice_sd_percent=float(np.std([r['evaluation']['dice'] for r in selected],ddof=1)*100) if len(selected)>1 else None,
            author_iou_percent=iou*100 if iou else None,author_dice_percent=dice*100 if dice else None,
            note='Author EAT table has IoU>Dice and is internally inconsistent; paper training hyperparameters unverified, official-code fallback accepted by user.' if model=='eat' else '',
            recipe=study.native.POLICY.get(model,{})))
    result=dict(status='completed' if len(completed)==12 else 'running',runs=rows,summary=summary,
        fixed=scope['fixed'],deferred=scope['deferred'])
    study.write(study.OUT/'comparison.json',result)
    columns=['model','seeds','iou_mean_percent','iou_sd_percent','dice_mean_percent','dice_sd_percent','author_iou_percent','author_dice_percent','note']
    with (study.OUT/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows({k:r[k] for k in columns} for r in summary)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args(argv)
    study.OUT.mkdir(parents=True,exist_ok=True)
    lock=(study.OUT/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (args.preflight or args.run):
        print(dict(models=study.MODELS,seeds=study.SEEDS,maximum_epochs_per_stage=100,patience_per_stage=10));return
    try:
        with (study.OUT/'test_paper_recipes.log').open('w') as stream:
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_paper_recipes.py','-v'],
                cwd=study.ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        smokes={}
        for model in study.MODELS:
            path=find('smoke',model,1234) or execute('smoke',model,1234)
            smokes[model]=str(path)
        study.write(study.OUT/'gate.json',dict(status='passed',signature=study.signature(),smokes=smokes))
        print('ALL_NATIVE_PREFLIGHT_PASSED',flush=True)
        if not args.run:
            return
        scope=study.read(study.OUT/'scope.json');scope['formal_rerun_started']=True
        study.write(study.OUT/'scope.json',scope)
        for model in study.MODELS:
            for seed in study.SEEDS:
                find('train',model,seed) or execute('train',model,seed)
                aggregate()
        result=aggregate()
        study.write(study.OUT/'progress.json',dict(status=result['status'],supervisor_pid=os.getpid(),completed_paper_runs=12))
    except BaseException as exc:
        study.write(study.OUT/'supervisor_failure.json',dict(error=repr(exc)))
        raise


if __name__=='__main__':
    main()
