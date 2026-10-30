"""Preview or run original experiment sources with official880/120 and FP32 budgets."""
import argparse
import ast
import csv
import difflib
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import statistics
import sys

from build_experiment import ROOT, build, verify_base

SPLIT = ROOT / 'configs/splits/kvasir_medico_official_880_120_v1'
HOOKS = Path(__file__).with_name('official_kvasir_hooks.py')
ORIGINALS = ('exp00_baseline', 'exp03_naf_p12', 'exp09_naf_discrete')
PAIRS = {
    'strict-v2': ('exp03_naf_p12_paired_strict_v2', 'exp09_naf_discrete_paired_strict_v2'),
    'existing-v1': ('exp03_1_naf_p12_paired_strict_v1', 'exp09_1_naf_discrete_paired_strict_v1'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def verify_split(dataset):
    manifest = json.loads((SPLIT / 'manifest.json').read_text())
    ids = {}
    for name in ('train', 'val', 'test'):
        p = SPLIT / (name + '.txt')
        if sha(p) != manifest['files'][p.name]['sha256']:
            raise RuntimeError(f'Official split changed: {p}')
        ids[name] = p.read_text().splitlines()
        if len(ids[name]) != len(set(ids[name])):
            raise RuntimeError('Duplicate split member')
    if len(ids['train']) != 880 or len(ids['val']) != 120 or ids['val'] != ids['test']:
        raise RuntimeError('Expected official880/120 with shared validation/reporting120')
    if set(ids['train']) & set(ids['val']):
        raise RuntimeError('Training/validation overlap')
    missing = [str(dataset / part / (n + '.jpg')) for n in ids['train'] + ids['val']
               for part in ('images', 'masks') if not (dataset / part / (n + '.jpg')).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing dataset files: {missing[:5]}')
    return manifest


def overlay(source):
    """Add one epoch-boundary budget hook; retain all existing model/input/loss code."""
    p = source / 'kvasir_train.py'
    old = p.read_text()
    needle = '        print()\n\n    best_loss_iou ='
    if old.count(needle) != 1:
        raise RuntimeError('Original epoch boundary changed; inspect instead of guessing')
    new = old.replace(needle, '        print()\n'
        '        if after_epoch(epoch, accuracy_list, loss_list, model, optimizer, scheduler, save_dir):\n'
        '            break\n\n    best_loss_iou =')
    loader_line = '    dataloaders = {"train": train_loader, "valid": val_loader}\n'
    if new.count(loader_line) != 1:
        raise RuntimeError('Original dataloader assignment changed')
    new = new.replace(loader_line, loader_line + '    dataloaders = budget_loaders(dataloaders)\n')
    new = 'from official_kvasir_hooks import configure_fp32, after_epoch, budget_loaders\nconfigure_fp32()\n' + new
    p.write_text(new)
    patch = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                    fromfile='a/kvasir_train.py', tofile='b/kvasir_train.py'))
    p = source / 'kvasir_test.py'
    old = p.read_text()
    new = ('from official_kvasir_hooks import configure_fp32, record_evaluation\n'
           'configure_fp32()\n') + old + '\n    record_evaluation(metrics)\n'
    p.write_text(new)
    patch += list(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                     fromfile='a/kvasir_test.py', tofile='b/kvasir_test.py'))
    shutil.copy2(HOOKS, source / HOOKS.name)
    for p in source.rglob('*.py'):
        ast.parse(p.read_text())
    (source.parent / 'official_budget.patch').write_text(''.join(patch))


def options(command, values):
    for k, v in values.items():
        command.append('--' + k)
        if v is not None:
            command.append(str(v))
    return command


def commands(experiment, seed, records, args):
    source, weights = records / 'source', records / 'checkpoints'
    cfg = json.loads((ROOT / 'experiments' / experiment / 'config.json').read_text())
    train_opts = dict(cfg['train_options'], epoch=args.epochs, hiera_checkpoint=str(args.hiera_checkpoint))
    train = [sys.executable, '-u', str(source / 'kvasir_train.py'), '--dataset', str(args.dataset),
        '--train_txt', str(SPLIT / 'train.txt'), '--val_txt', str(SPLIT / 'val.txt'),
        '--output_dir', str(weights), '--log_dir', str(records / 'logs'), '--seed', str(seed)]
    options(train, train_opts)
    evaluation = [sys.executable, '-u', str(source / 'kvasir_test.py'), '--dataset', str(args.dataset),
        '--split_txt', str(SPLIT / 'val.txt'), '--model', str(weights / 'kvasir_best_model.pth'),
        '--debug_dir', str(records / 'visualization')]
    options(evaluation, cfg['eval_options'])
    return cfg, train, evaluation


def execute(experiment, seed, records, args):
    records.mkdir(parents=True, exist_ok=False)
    snapshot = build(experiment, records / 'source')
    overlay(records / 'source')
    cfg, train, evaluation = commands(experiment, seed, records, args)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
        NO_ALBUMENTATIONS_UPDATE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED=str(seed),
        OFFICIAL_KVASIR_RECORDS=str(records), OFFICIAL_KVASIR_PATIENCE=str(args.patience),
        OFFICIAL_KVASIR_MAX_EPOCHS=str(args.epochs),
        OFFICIAL_KVASIR_SMOKE_STEPS=str(args.smoke_steps if args.smoke_only else 0))
    manifest = dict(status='prepared', experiment=experiment, seed=seed, config=cfg,
        training_command=train, evaluation_command=evaluation, source=snapshot,
        split=json.loads((SPLIT / 'manifest.json').read_text()),
        hiera_sha256=sha(args.hiera_checkpoint), precision='FP32; AMP and TF32 disabled',
        stopping=dict(max_epochs=args.epochs, patience=args.patience, metric='validation IoU',
                      min_delta=0, tie='no improvement; original best-checkpoint tie policy retained'),
        source_files={p.relative_to(records / 'source').as_posix():sha(p)
            for p in (records / 'source').rglob('*') if p.is_file()},
        effective_train_overrides=dict(epoch=args.epochs, hiera_checkpoint=str(args.hiera_checkpoint)),
        historical_results='not modified', independent_test=False)
    manifest['run_kind'] = 'preflight' if args.smoke_only else 'formal'
    write(records / 'manifest.json', manifest)
    (records / 'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True, env=env))
    try:
        for command, name in ((train, 'train'), (evaluation, 'evaluation')):
            manifest['status'] = name
            write(records / 'manifest.json', manifest)
            print('START', experiment, seed, name, flush=True)
            with (records / (name + '.console.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=records / 'source', env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
                manifest['child_pid'] = child.pid
                write(records / 'manifest.json', manifest)
                def stop(signum, frame):
                    if child.poll() is None:
                        child.send_signal(signum)
                    raise SystemExit(128 + signum)
                handlers = {sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
                try:
                    code = child.wait()
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill();child.wait()
                    for sig,handler in handlers.items():signal.signal(sig,handler)
            if code:
                raise RuntimeError(f'{name} failed ({code}); see {records}')
        weights = sorted((records / 'checkpoints').glob('*.pth'))
        if len(weights) != 3:
            raise RuntimeError(f'Expected three original checkpoints, found {len(weights)}')
        manifest.update(status='completed', checkpoint_files={p.name:sha(p) for p in weights},
            evaluation=json.loads((records / 'evaluation.json').read_text()))
        progress = json.loads((records / 'progress.json').read_text())
        progress.update(status='completed', evaluation=manifest['evaluation']['metrics'])
        write(records / 'progress.json', progress)
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        write(records / 'manifest.json', manifest)
    return dict(experiment=experiment, seed=seed, records=str(records),
                completed_epochs=progress['completed_epochs'], metrics=manifest['evaluation']['metrics'])


def summarize(directory, completed):
    rows = []
    for experiment in dict.fromkeys(r['experiment'] for r in completed):
        runs = [r for r in completed if r['experiment'] == experiment]
        values = dict(experiment=experiment, seeds=len(runs),
            seed_ids=[r['seed'] for r in runs], epochs=[r['completed_epochs'] for r in runs])
        for metric in ('iou','dice','precision','recall'):
            scores = [r['metrics'][metric] * 100 for r in runs]
            values[metric + '_mean_percent'] = statistics.mean(scores)
            values[metric + '_sd_percent'] = statistics.stdev(scores) if len(scores) > 1 else None
        rows.append(values)
    write(directory / 'summary.json', dict(status='completed' if len(completed)==18 else 'partial',
        comparison_scope='Official880/120; same120 for validation selection and reporting; no independent test',
        summary=rows, runs=completed))
    if rows:
        with (directory / 'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in rows[0])+'</tr>' for r in rows) if rows else ''
    headings=''.join('<th>'+html.escape(k)+'</th>' for k in rows[0]) if rows else ''
    (directory / 'summary.html').write_text('<!doctype html><meta charset="utf-8"><title>Official Kvasir FP32</title>'
        '<style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}</style>'
        '<h1>Official Kvasir880/120 — FP32</h1><p>Mean and sample SD in %. Same120 used for early stopping and reporting. '
        'Original0/3, original9(paired-v1), and strict-v2 groups retain their respective input/numerical protocols.</p>'
        '<table><tr>'+headings+'</tr>'+body+'</table>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-protocol', choices=tuple(PAIRS), required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--hiera-checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seeds', nargs='+', type=int, choices=(1234,42,2026), default=[1234,42,2026])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--smoke-only', action='store_true', help='Three real train/valid batches per model; one epoch; no formal run')
    parser.add_argument('--smoke-steps', type=int, default=3)
    parser.add_argument('--run', action='store_true', help='Without this flag, validate and preview only')
    args = parser.parse_args()
    if min(args.epochs,args.patience,args.smoke_steps) < 1 or args.gpu < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Invalid budget/GPU or duplicate seeds')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.','..'):
        parser.error('run-name must be one directory name')
    for name in ('dataset','hiera_checkpoint','output_root'):
        setattr(args,name,getattr(args,name).resolve())
    verify_base()
    split = verify_split(args.dataset)
    if not args.hiera_checkpoint.is_file():
        raise FileNotFoundError(args.hiera_checkpoint)
    if sha(args.hiera_checkpoint) != '7694579e76566289f25f63799024a2fe83144005c75ece6c09458254aa3ee13c':
        raise RuntimeError('Official Hiera checkpoint checksum mismatch')
    experiments = (*ORIGINALS,'exp00_baseline_paired_strict_v2',*PAIRS[args.paired_protocol])
    if args.smoke_only:
        args.epochs = 1
        args.seeds = [1234]
    directory = args.output_root / args.run_name
    plan = dict(experiments=experiments, seeds=args.seeds, epochs=args.epochs, patience=args.patience,
        precision='FP32; AMP and TF32 disabled', split=split, output=str(directory),
        scheduler=f'Original CosineAnnealingLR; T_max follows epoch={args.epochs}',
        initial_weights='Fresh official Hiera for every run, no previous trained weights',
        scope='Six project experiments only; paper-model reruns require a separate confirmed queue')
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    for seed in args.seeds:
        for experiment in experiments:
            records = directory / f'{experiment}_seed{seed}'
            _, train, evaluation = commands(experiment,seed,records,args)
            print('TRAIN',shlex.join(train),flush=True)
            print('EVAL ',shlex.join(evaluation),flush=True)
    if not args.run:
        print('PREVIEW ONLY. No new training or output run directory created.',flush=True)
        return
    directory.mkdir(parents=True,exist_ok=False)
    lock=(directory / 'study.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(directory / 'plan.json',plan)
    completed=[]
    try:
        if not args.smoke_only:
            import copy
            preflight_args = copy.copy(args)
            preflight_args.smoke_only, preflight_args.epochs = True, 1
            for experiment in experiments:
                write(directory / 'progress.json',dict(status='preflight',experiment=experiment,
                    completed=completed,supervisor_pid=os.getpid()))
                execute(experiment,1234,directory/f'smoke_{experiment}_seed1234',preflight_args)
        for seed in args.seeds:
            for experiment in experiments:
                write(directory / 'progress.json',dict(status='running',experiment=experiment,
                    seed=seed,completed=completed,supervisor_pid=os.getpid()))
                completed.append(execute(experiment,seed,directory/f'{experiment}_seed{seed}',args))
                write(directory / 'results.json',completed)
                summarize(directory,completed)
        write(directory / 'progress.json',dict(status='completed',completed=completed))
    except BaseException as exc:
        write(directory / 'progress.json',dict(status='failed',error=str(exc),completed=completed))
        raise


if __name__ == '__main__':
    main()
