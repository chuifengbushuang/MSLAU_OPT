"""Summarize recorded NAF runs without checkpoints, GPU access, or training.

Duplicate experiment/seed pairs are rejected: this tool never selects the best
repeat. Original records are read only; reports contain source-file SHA256s.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

EXPERIMENTS = ('exp00_baseline', 'exp01_naf_all', 'exp02_naf_p34', 'exp03_naf_p12')
SEEDS = (1234, 42, 2026)
STAGES = {'exp00_baseline': (), 'exp01_naf_all': (1, 2, 3, 4),
          'exp02_naf_p34': (3, 4), 'exp03_naf_p12': (1, 2)}
METRIC = re.compile(r'^(train|valid) Loss: ([\d.]+) MainLoss: ([\d.]+) DistillLoss: ([\d.]+) IoU: ([\d.]+)$')
NAF = re.compile(r'^(train|valid) NAF P([1-4]): (.*)$')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_epochs(text):
    rows, stages = [], []
    epoch = None
    for line in text.splitlines():
        match = re.match(r'^Epoch (\d+)/(\d+)$', line)
        if match:
            epoch = int(match[1])
        match = METRIC.match(line)
        if match:
            phase, loss, main, distill, iou = match.groups()
            rows.append(dict(epoch=epoch, phase=phase, loss=float(loss), main_loss=float(main),
                             distill_loss=float(distill), iou=float(iou)))
        match = NAF.match(line)
        if match:
            phase, stage, values = match.groups()
            stages.append(dict(epoch=epoch, phase=phase, stage=int(stage),
                               **{key: float(value) for key, value in re.findall(r'(\w+)=([\d.eE+-]+)', values)}))
    return rows, stages


def average(values):
    values = list(values)
    return dict(mean=statistics.mean(values), sample_std=statistics.stdev(values),
                minimum=min(values), maximum=max(values))


def csv_file(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def analyze(input_root, output):
    input_root, output = Path(input_root).resolve(), Path(output).resolve()
    run_rows, epochs, stats_rows, evidence, documents = [], [], [], {}, {}
    seen = set()
    for record in sorted(input_root.glob('exp*/manifest.json')):
        saved = json.loads(record.read_text())
        experiment, seed = saved['config']['id'], saved['seed']
        if experiment not in EXPERIMENTS:
            continue
        pair = experiment, seed
        if pair in seen:
            raise ValueError(f'Duplicate experiment/seed; choose a documented batch explicitly: {pair}')
        seen.add(pair)
        if seed not in SEEDS or saved['status'] != 'completed' or saved['source']['worktree_dirty']:
            raise ValueError(f'Incomplete or uncontrolled run: {record}')
        directory = record.parent
        required = ('manifest.json', 'console.log', 'evaluation.log', 'environment.txt', 'runtime.txt')
        for name in required:
            path = directory / name
            evidence[str(path.relative_to(input_root))] = digest(path)
        text = (directory / 'console.log').read_text()
        epoch_rows, stage_rows = parse_epochs(text)
        for phase in ('train', 'valid'):
            if [row['epoch'] for row in epoch_rows if row['phase'] == phase] != list(range(20)):
                raise ValueError(f'Expected exactly 20 epochs for {phase}: {record}')
        if len(saved['checkpoint_files']) != 3:
            raise ValueError(f'Expected three recorded checkpoint files: {record}')
        match = re.search(r'Best val IoU: ([\d.]+) at epoch (\d+)', text)
        if not match or float(match[1]) != saved['best_val_iou'] or int(match[2]) != saved['best_epoch']:
            raise ValueError(f'Best metric disagrees with manifest: {record}')
        evaluation = {key: float(value) for key, value in re.findall(
            r'^mean ([\w-]+): ([\d.]+)', (directory / 'evaluation.log').read_text(), re.M)}
        if evaluation != saved['evaluation_metrics'] or abs(evaluation['IoU'] - saved['best_val_iou']) > 1e-4:
            raise ValueError(f'Independent evaluation mismatch: {record}')
        config = saved['config']
        for options in ('train_options', 'eval_options'):
            expected = ','.join(map(str, STAGES[experiment]))
            if config[options].get('naf_stages', '') != expected:
                raise ValueError(f'Wrong stages in {record}')
        for row in epoch_rows:
            epochs.append(dict(experiment=experiment, seed=seed, **row))
        for row in stage_rows:
            if row['stage'] not in STAGES[experiment] or abs(row['keep_all'] - 0.9) > 2e-6:
                raise ValueError(f'Stage/budget mismatch: {record}')
            stats_rows.append(dict(experiment=experiment, seed=seed, **row))
        for epoch in range(20):
            for phase in ('train', 'valid'):
                present = [row['stage'] for row in stage_rows if row['epoch'] == epoch and row['phase'] == phase]
                if tuple(present) != STAGES[experiment]:
                    raise ValueError(f'Missing/duplicate diagnostics: {record}')
        timing = re.search(r'Training complete in (\d+)m (\d+)s', text)
        fps = re.search(r'^FPS: ([\d.]+)', (directory / 'evaluation.log').read_text(), re.M)
        row = dict(experiment=experiment, seed=seed, run_id=saved['run_id'],
                   best_val_iou=saved['best_val_iou'], best_epoch=saved['best_epoch'],
                   **{'eval_' + key: value for key, value in evaluation.items()},
                   train_seconds=int(timing[1]) * 60 + int(timing[2]), eval_fps=float(fps[1]),
                   gpu=saved['gpu'], git_commit=saved['source']['repository_commit'],
                   environment_sha256=digest(directory / 'environment.txt'),
                   runtime_sha256=digest(directory / 'runtime.txt'),
                   hiera_sha256=saved['hiera_sha256'])
        run_rows.append(row)
        documents[pair] = saved
    if seen != {(experiment, seed) for experiment in EXPERIMENTS for seed in SEEDS}:
        raise ValueError('Missing runs: expected exactly three seeds in each of experiments 0–3')
    # Verify the actual recorded source/config comparisons, not only folder names.
    repository = Path(__file__).resolve().parents[1]
    frozen = json.loads((repository / 'experiments/exp00_baseline/source_manifest.json').read_text())['files']
    for seed in SEEDS:
        for name, expected in frozen.items():
            if documents['exp00_baseline', seed]['source']['files'][name] != expected:
                raise ValueError(f'Baseline source differs from frozen original: {name}')
    patch_hash = digest(repository / 'experiments/shared_naf_v1/changes.patch')
    module_hash = digest(repository / 'experiments/shared_naf_v1/added/networks/naf_select_reg.py')
    for experiment in EXPERIMENTS[1:]:
        for seed in SEEDS:
            source = documents[experiment, seed]['source']
            if source['patch_sha256'] != patch_hash or source['added_files']['networks/naf_select_reg.py'] != module_hash:
                raise ValueError('Recorded NAF differs from the analyzed shared_naf_v1 source')
    for seed in SEEDS:
        baseline = documents['exp00_baseline', seed]
        for experiment in EXPERIMENTS:
            saved = documents[experiment, seed]
            for category in ('train_options', 'eval_options'):
                options = {k: v for k, v in saved['config'][category].items() if k not in {'naf_stages', 'naf_reg_weight'}}
                if options != baseline['config'][category]:
                    raise ValueError(f'Other training/evaluation settings changed: {experiment}')
            if experiment != 'exp00_baseline' and saved['config']['train_options']['naf_reg_weight'] != 0.001:
                raise ValueError('Unexpected regularizer coefficient')
            for name in ('loader.py', 'loss.py', 'configs/splits/kvasir/train.txt', 'configs/splits/kvasir/val.txt'):
                if saved['source']['files'][name] != baseline['source']['files'][name]:
                    raise ValueError(f'Unmatched original file: {name}')
    for key in ('git_commit', 'environment_sha256', 'runtime_sha256', 'hiera_sha256', 'gpu'):
        if len({row[key] for row in run_rows}) != 1:
            raise ValueError(f'Mixed experimental conditions: {key}')
    if len({documents[experiment, seed]['source']['patch_sha256']
            for experiment in EXPERIMENTS[1:] for seed in SEEDS}) != 1:
        raise ValueError('NAF variants did not use identical patch code')
    if len({documents[experiment, seed]['source']['added_files']['networks/naf_select_reg.py']
            for experiment in EXPERIMENTS[1:] for seed in SEEDS}) != 1:
        raise ValueError('NAF variants did not use identical added module code')
    summaries, deltas = {}, []
    lookup = {(row['experiment'], row['seed']): row for row in run_rows}
    for experiment in EXPERIMENTS:
        rows = [lookup[experiment, seed] for seed in SEEDS]
        difference = [row['best_val_iou'] - lookup['exp00_baseline', row['seed']]['best_val_iou'] for row in rows]
        summaries[experiment] = dict(iou=average(row['best_val_iou'] for row in rows),
            dice=average(row['eval_dice'] for row in rows),
            precision=average(row['eval_precision'] for row in rows),
            recall=average(row['eval_recall'] for row in rows),
            delta_vs_baseline=average(difference), wins=sum(value > 0 for value in difference),
            train_seconds=average(row['train_seconds'] for row in rows),
            metric_source='best validation IoU: 6 decimals; independent evaluation metrics: 4 decimals')
        for row, delta in zip(rows, difference):
            deltas.append(dict(experiment=experiment, seed=row['seed'], best_val_iou=row['best_val_iou'],
                               baseline_iou=lookup['exp00_baseline', row['seed']]['best_val_iou'], delta=delta,
                               delta_percentage_points=100 * delta))
    best_stats = []
    for row in stats_rows:
        saved = documents[row['experiment'], row['seed']]
        if row['phase'] == 'valid' and row['epoch'] == saved['best_epoch']:
            best_stats.append(dict(**row,
                effective_low=1 - row['alpha'] + row['alpha'] * row['keep_low'],
                effective_mid=1 - row['alpha'] + row['alpha'] * row['keep_mid'],
                effective_high=1 - row['alpha'] + row['alpha'] * row['keep_high']))
    summary = dict(input_root=str(input_root), run_count=len(run_rows),
                   evidence_sha256=evidence, experiments=summaries,
                   caveats=['Three seeds per setting; exploratory comparison, no significance claim.',
                            'Same seed does not fix Albumentations trajectories in the frozen code.',
                            'Validation selects checkpoints; no independent held-out test set used.',
                            'Timing is observational under shared server load, not a controlled benchmark.',
                            'Weights not supplied: no weight contents, per-image masks or predictions inspected.'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    csv_file(output / 'runs.csv', run_rows)
    csv_file(output / 'seed_deltas.csv', deltas)
    csv_file(output / 'epochs.csv', epochs)
    csv_file(output / 'naf_diagnostics.csv', stats_rows)
    csv_file(output / 'naf_at_best_epoch.csv', best_stats)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == '__main__':
    main()
