"""Audit F/S runs, compare curves, and optionally reevaluate all four checkpoints.

The probe uses eval/inference_mode only. It never updates or writes checkpoints.
"""
import argparse
import ast
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import fredft_study as driver
import local_decoder_study as common

ROOT = driver.WORKSPACE
OUT = ROOT / 'analysis/20261028_fredft_study'
FIG = ROOT / 'visualization/local_fredft_study_20261028'
LABELS = dict(baseline='Original baseline', naf_decoder='NAF + original decoder',
              fredft_frequency='F: MFDA + FDFFL', fredft_spatial='S: spatial control')


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_runs(seed=1234):
    runs = dict(driver.OLD_RUNS) if seed == 1234 else {}
    for variant in ('fredft_frequency', 'fredft_spatial'):
        found = [p for p in (ROOT / 'experiments').glob(f'local_train_{variant}_seed{seed}_*')
                 if (p / 'result.json').is_file()]
        if len(found) != 1:
            raise RuntimeError(f'Expected one completed {variant}/seed{seed}, got {found}')
        runs[variant] = found[0]
    return runs


def comparable_ast(path, name):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    class StripSkip(ast.NodeTransformer):
        def visit_If(self, n):
            if isinstance(n.test, ast.Compare) and 'naf_replacement_skip' in ast.unparse(n.test):
                return None
            return self.generic_visit(n)
    return ast.dump(StripSkip().visit(node), include_attributes=False)


def summarize(runs):
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    reference = read(runs['baseline'] / 'initial_state.json')
    inputs = lines(runs['baseline'] / 'input_fingerprints.jsonl')
    config = read(runs['baseline'] / 'effective_config.json')
    checks, rows, histories = {}, [], {}
    for variant, records in runs.items():
        result, initial = read(records / 'result.json'), read(records / 'initial_state.json')
        effective, manifest = read(records / 'effective_config.json'), read(records / 'manifest.json')
        history = lines(records / 'epoch_metrics.jsonl')
        check = dict(completed=result['status'] == 'completed' and len(history) == 20,
            shared_initial_state=initial['shared_state_sha256'] == reference['shared_state_sha256'],
            cpu_rng=initial['cpu_rng_sha256'] == reference['cpu_rng_sha256'],
            environment=(records / 'environment.txt').read_text() == (runs['baseline'] / 'environment.txt').read_text(),
            all40_inputs=lines(records / 'input_fingerprints.jsonl') == inputs,
            common_configuration={k: v for k, v in effective.items() if k not in ('variant', 'naf_reg_weight')} ==
                                 {k: v for k, v in config.items() if k not in ('variant', 'naf_reg_weight')},
            best_reloaded=abs(result['evaluation']['iou'] - result['best_iou']) < 1e-6,
            model_source=common.file_hash(records / 'source/networks/mslau_net.py') ==
                         common.file_hash(runs['baseline'] / 'source/networks/mslau_net.py'),
            paired_source=common.file_hash(records / 'source/paired_inputs.py') ==
                          common.file_hash(runs['baseline'] / 'source/paired_inputs.py'),
            three_checkpoints=len(result['checkpoints']) == 3,
            original_loop_ast=all(comparable_ast(records / ('common_driver.py' if variant.startswith('fredft_') else 'driver.py'), name) ==
                                  comparable_ast(runs['baseline'] / 'driver.py', name)
                                  for name in ('train', 'make_optimizer', 'autocast')))
        for name, expected in result['checkpoints'].items():
            actual = common.file_hash(ROOT / 'checkpoints' / records.name / name)
            check['checkpoint_' + name] = actual == expected
        if variant.startswith('fredft_'):
            check['snapshot_core'] = common.file_hash(records / 'source/networks/fredft_core.py') == manifest['core_sha256']
            check['snapshot_driver'] = common.file_hash(records / 'driver.py') == manifest['driver_sha256']
            check['snapshot_common_driver'] = common.file_hash(records / 'common_driver.py') == manifest['common_driver_sha256']
        if not all(check.values()):
            raise RuntimeError(f'Audit failed for {variant}: {check}')
        checks[variant] = check
        rows.append(dict(variant=variant, label=LABELS[variant], iou=result['evaluation']['iou'],
            dice=result['evaluation']['dice'], precision=result['evaluation']['precision'],
            recall=result['evaluation']['recall'], best_epoch=result['best_epoch'] + 1,
            last5_iou_mean=statistics.mean(x['valid']['iou'] for x in history[-5:]),
            last5_iou_sd=statistics.stdev(x['valid']['iou'] for x in history[-5:]),
            epoch20_train_main_loss=history[-1]['train']['main_loss'],
            epoch20_valid_main_loss=history[-1]['valid']['main_loss'],
            elapsed_minutes=result['elapsed_minutes'], trainable_parameters=initial['trainable_parameters'],
            records=str(records), checkpoint=str(ROOT / 'checkpoints' / records.name / 'kvasir_best_model.pth')))
        histories[variant] = history
    f_initial, s_initial = (read(runs[k] / 'initial_state.json') for k in ('fredft_frequency', 'fredft_spatial'))
    checks['fs_complete_initial_state'] = f_initial == s_initial
    checks['fs_environment'] = (runs['fredft_frequency'] / 'environment.txt').read_text() == (runs['fredft_spatial'] / 'environment.txt').read_text()
    if not checks['fs_complete_initial_state'] or not checks['fs_environment']:
        raise RuntimeError('F/S initial state or environment mismatch')
    values = {r['variant']: r['iou'] for r in rows}
    for row in rows:
        row['delta_baseline_pp'] = 100 * (row['iou'] - values['baseline'])
        row['delta_naf_pp'] = 100 * (row['iou'] - values['naf_decoder'])
        row['delta_spatial_pp'] = 100 * (row['iou'] - values['fredft_spatial'])
    decision = values['fredft_frequency'] > max(values['baseline'], values['fredft_spatial'])
    summary = dict(seed=1234, independent_test_set=False, checks=checks, runs=rows,
                   expand_multiseed=decision, selection='Best validation IoU under equal20-epoch budget',
                   frequency_contrast_caveat='FDFFL FFT/space are mathematically equivalent; native TF32 amplifies small FFT roundoff. See smoke_audit.json.')
    common.write_json(OUT / 'summary.json', summary)
    write_csv(OUT / 'runs.csv', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    colors = dict(baseline='#475569', naf_decoder='#94a3b8', fredft_frequency='#dc2626', fredft_spatial='#2563eb')
    for row in rows:
        variant = row['variant']; h = histories[variant]
        epochs = [x['epoch'] + 1 for x in h]
        axes[0].plot(epochs, [x['valid']['iou'] for x in h], color=colors[variant], label=LABELS[variant])
        axes[0].scatter([row['best_epoch']], [row['iou']], color=colors[variant], marker='*', s=100)
        axes[1].plot(epochs, [x['valid']['main_loss'] for x in h], color=colors[variant])
        axes[2].plot(epochs, [x['train']['main_loss'] for x in h], color=colors[variant])
    for ax in axes:
        ax.set_xlabel('Epoch');ax.set_xticks((1, 5, 10, 15, 20));ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean per-image validation IoU');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Validation main Structure loss');axes[2].set_ylabel('Training main Structure loss')
    fig.suptitle('FreDFT core study | seed1234 | BF16 | paired_aug_v2 | original decoder')
    fig.savefig(FIG / 'learning_curves.png', dpi=180);plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def probe(runs):
    import torch
    import numpy as np
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from scipy.ndimage import distance_transform_edt
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True;torch.backends.cudnn.benchmark = False
    import cv2
    cv2.setNumThreads(0)
    source = runs['baseline'] / 'source'
    sys.path.insert(0, str(source))
    import paired_inputs as paired
    factories = common.definitions(source / 'kvasir_train.py', {'load_ids', 'get_valid_transform'},
                                   dict(A=A, ToTensorV2=ToTensorV2, os=os))
    factories.update(common.definitions(source / 'kvasir_test.py', {'compute_batch_metrics'}, dict(torch=torch)))
    ids = factories['load_ids'](str(source / 'configs/splits/kvasir/val.txt'))
    dataset = paired.PairedDataset(str(ROOT / 'datasets/Kvasir-SEG'), ids,
                                  factories['get_valid_transform'](352), 1234, 'valid')
    loader = paired.make_loader(dataset, 6, 2, pin_memory=True)
    per_image, outputs = [], {}
    for variant, records in runs.items():
        arguments = read(records / 'arguments.json')
        args = argparse.Namespace(**arguments)
        args.hiera_checkpoint = Path(args.hiera_checkpoint)
        factory = driver.make_model if variant.startswith('fredft_') else common.make_model
        model = factory(records / 'source', args)
        checkpoint = ROOT / 'checkpoints' / records.name / 'kvasir_best_model.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True);del state
        model.requires_grad_(False).cuda().eval()
        rows, predictions = [], []
        digest = paired.InputDigest()
        with torch.inference_mode(), common.autocast(args):
            for images, masks, image_ids in loader:
                digest.update(images, masks, image_ids)
                logits = model(images.cuda(non_blocking=True))
                prediction = logits.float().sigmoid() > .5
                metrics = factories['compute_batch_metrics'](prediction.float(), masks.cuda(non_blocking=True).unsqueeze(1))
                pred = prediction[:, 0].cpu().numpy();gt = masks.numpy() > 0
                for j, image_id in enumerate(image_ids):
                    band = np.where(gt[j], distance_transform_edt(gt[j]), distance_transform_edt(~gt[j])) <= 4
                    error = pred[j] != gt[j]
                    intersection = np.count_nonzero(pred[j] & gt[j] & band)
                    union = np.count_nonzero((pred[j] | gt[j]) & band)
                    row = dict(variant=variant, id=image_id, **{k: v[j] for k, v in metrics.items()},
                        gt_area_fraction=float(gt[j].mean()), errors=int(error.sum()),
                        errors_4px_band=int((error & band).sum()),
                        false_positive_far=int((pred[j] & ~gt[j] & ~band).sum()),
                        false_negative_far=int((~pred[j] & gt[j] & ~band).sum()),
                        band_iou=(intersection + 1) / (union + 1))
                    rows.append(row)
                predictions.extend(pred)
        expected_digest = next(x for x in lines(records / 'input_fingerprints.jsonl') if x['phase'] == 'valid' and x['epoch'] == 0)
        assert digest.record(1234, 0, 'valid') == expected_digest
        evaluation = {k: statistics.mean(row[k] for row in rows) for k in ('iou', 'dice', 'precision', 'recall')}
        expected = read(records / 'result.json')['evaluation']
        assert all(abs(evaluation[k] - expected[k]) < 1e-6 for k in evaluation), (variant, evaluation, expected)
        outputs[variant] = dict(**evaluation, band_iou_mean=statistics.mean(r['band_iou'] for r in rows),
            total_errors=sum(r['errors'] for r in rows), total_errors_4px_band=sum(r['errors_4px_band'] for r in rows),
            false_positive_far=sum(r['false_positive_far'] for r in rows),
            false_negative_far=sum(r['false_negative_far'] for r in rows),
            checkpoint=str(checkpoint), original_metrics_reproduced=True, validation_inputs_reproduced=True)
        per_image.extend(rows)
        np.savez_compressed(OUT / f'predictions_{variant}.npz', ids=np.asarray(ids), masks=np.asarray(predictions, dtype=np.uint8))
        print('PROBE', variant, json.dumps(outputs[variant]), flush=True)
        del model, rows, predictions, prediction, logits
        gc.collect();torch.cuda.empty_cache()
    baseline_rows = {r['id']: r for r in per_image if r['variant'] == 'baseline'}
    spatial_rows = {r['id']: r for r in per_image if r['variant'] == 'fredft_spatial'}
    for row in per_image:
        row['delta_baseline_iou'] = row['iou'] - baseline_rows[row['id']]['iou']
        row['delta_spatial_iou'] = row['iou'] - spatial_rows[row['id']]['iou']
    for variant in runs:
        selected = [r for r in per_image if r['variant'] == variant]
        for reference in ('baseline', 'spatial'):
            delta = [r[f'delta_{reference}_iou'] for r in selected]
            outputs[variant][f'wins_vs_{reference}'] = sum(x > 1e-8 for x in delta)
            outputs[variant][f'losses_vs_{reference}'] = sum(x < -1e-8 for x in delta)
    write_csv(OUT / 'per_image.csv', per_image)
    common.write_json(OUT / 'probe.json', dict(mode='inference_only', optimizer_updates=0,
        boundary_band='Euclidean distance<=4pixels to the opposite GT class at352x352',
        caveat='Validation checkpoints selected on these120images; no independent test or causal texture attribution.', runs=outputs))
    # Fixed selection rule: biggest F improvements/degradations versus baseline.
    ranked = sorted((r for r in per_image if r['variant'] == 'fredft_frequency'), key=lambda r: r['delta_baseline_iou'])
    selected_ids = [r['id'] for r in ranked[:3] + ranked[-3:]]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    predictions = {v: np.load(OUT / f'predictions_{v}.npz')['masks'] for v in runs}
    fig, axes = plt.subplots(6, 6, figsize=(16, 16), constrained_layout=True)
    for row_index, image_id in enumerate(selected_ids):
        index = ids.index(image_id)
        image = cv2.cvtColor(cv2.resize(cv2.imread(str(ROOT / 'datasets/Kvasir-SEG/images' / image_id)), (352, 352)), cv2.COLOR_BGR2RGB)
        gt = dataset[index][1].numpy() > 0
        axes[row_index, 0].imshow(image);axes[row_index, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        for col, variant in enumerate(runs, 2):
            pred = predictions[variant][index].astype(bool)
            overlay = np.zeros((352, 352, 3), dtype=np.uint8)
            overlay[gt & pred] = (230, 230, 230)
            overlay[~gt & pred] = (245, 90, 70)
            overlay[gt & ~pred] = (60, 140, 245)
            axes[row_index, col].imshow(overlay)
        for col in range(6):
            axes[row_index, col].set_xticks([]);axes[row_index, col].set_yticks([])
        delta = next(r['delta_baseline_iou'] for r in ranked if r['id'] == image_id)
        axes[row_index, 0].set_ylabel(f'{image_id}\nF-base:{100*delta:+.2f}pp', fontsize=7)
    for col, title in enumerate(['Input', 'GT'] + [LABELS[v] for v in runs]):
        axes[0, col].set_title(title, fontsize=9)
    fig.suptitle('Validation examples |3 largest F regressions then3 improvements | white TP / red FP / blue FN')
    fig.savefig(FIG / 'error_examples.png', dpi=130);plt.close(fig)


def summarize_confirmation():
    import confirm_fredft_study as confirmation
    runs = find_runs()
    rows, checks = [], []
    for seed in (1234, 42, 2026):
        records = ({k: runs[k] for k in ('baseline', 'fredft_frequency', 'fredft_spatial')}
                   if seed == 1234 else {k: confirmation.completed(k, seed)
                                        for k in ('baseline', 'fredft_frequency', 'fredft_spatial')})
        if any(path is None for path in records.values()):
            raise RuntimeError(f'Confirmation incomplete: {seed}')
        if seed != 1234:
            checks.append(confirmation.validate_seed(seed, records))
        results = {key: read(path / 'result.json') for key, path in records.items()}
        for variant, result in results.items():
            rows.append(dict(seed=seed, variant=variant, **result['evaluation'],
                best_epoch=result['best_epoch'] + 1, elapsed_minutes=result['elapsed_minutes'],
                delta_baseline_pp=100 * (result['evaluation']['iou'] - results['baseline']['evaluation']['iou']),
                delta_spatial_pp=100 * (result['evaluation']['iou'] - results['fredft_spatial']['evaluation']['iou']),
                records=str(records[variant])))
    aggregates = {}
    for variant in ('baseline', 'fredft_frequency', 'fredft_spatial'):
        selected = [r for r in rows if r['variant'] == variant]
        aggregates[variant] = {metric + '_mean': statistics.mean(r[metric] for r in selected)
                               for metric in ('iou', 'dice', 'precision', 'recall', 'delta_baseline_pp', 'delta_spatial_pp')}
        aggregates[variant]['iou_sample_sd'] = statistics.stdev(r['iou'] for r in selected)
        aggregates[variant]['wins_vs_baseline'] = sum(r['delta_baseline_pp'] > 0 for r in selected)
        aggregates[variant]['wins_vs_spatial'] = sum(r['delta_spatial_pp'] > 0 for r in selected)
    summary = dict(seeds=[1234, 42, 2026], aggregates=aggregates, runs=rows, audits=checks,
        independent_test_set=False, caveat='Three preselected seeds; descriptive paired results on the same120validation images. Not a statistical significance or independent generalization claim.')
    common.write_json(OUT / 'confirmation_summary.json', summary)
    write_csv(OUT / 'confirmation_runs.csv', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for seed, color in zip((1234, 42, 2026), ('#dc2626', '#2563eb', '#16a34a')):
        selected = [r for r in rows if r['seed'] == seed]
        axes[0].plot(range(3), [r['iou'] for r in selected], marker='o', color=color, label=f'seed{seed}')
        f = next(r for r in selected if r['variant'] == 'fredft_frequency')
        axes[1].plot(range(2), [f['delta_baseline_pp'], f['delta_spatial_pp']], marker='o', color=color, label=f'seed{seed}')
    axes[0].set_xticks(range(3), ('Baseline', 'F: frequency', 'S: spatial'))
    axes[0].set_ylabel('Best-checkpoint validation IoU')
    axes[1].set_xticks(range(2), ('F - baseline', 'F - S'))
    axes[1].set_ylabel('Paired IoU difference (percentage points)');axes[1].axhline(0, color='#64748b', ls='--')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same-machine three-seed confirmation | BF16 |20epochs each')
    fig.savefig(FIG / 'confirmation.png', dpi=180);plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    runs = find_runs()
    summarize(runs)
    if args.probe:
        probe(runs)
    if args.confirmation:
        summarize_confirmation()
