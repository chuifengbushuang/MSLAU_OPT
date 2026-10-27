"""Inference-only diagnosis of completed strict-v2 baseline or experiment-9 checkpoints.

Uses the frozen evaluator's loader, preprocessing and metrics, not a new model.
All masks/metrics are measured on the original evaluation grid (352 x 352).
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from build_experiment import ROOT, build

EXPERIMENT = 'exp00_baseline_paired_strict_v2'
EXPERIMENTS = (EXPERIMENT, 'exp09_naf_discrete_paired_strict_v2')
RADII = (2, 4, 8)
SIZE_BINS = ('<1%', '1-5%', '5-20%', '>=20%')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def size_bin(fraction):
    return SIZE_BINS[sum(fraction >= edge for edge in (.01, .05, .20))]


def boundary(mask):
    binary = np.asarray(mask, dtype=np.uint8)
    # Outside the image is background, including for objects touching its edge.
    inner = cv2.erode(binary, np.ones((3, 3), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (binary - inner).astype(bool)


def distance_to(points):
    if not points.any():
        return np.full(points.shape, np.inf)
    return cv2.distanceTransform((~points).astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE)


def diagnose(prediction, target):
    """Geometry only: these flags do not identify texture/noise causally."""
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.ndim != 2 or pred.shape != gt.shape:
        raise ValueError('Expected two equally sized 2-D masks')
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
    pred_edge, gt_edge = boundary(pred), boundary(gt)
    gt_distance, pred_distance = distance_to(gt_edge), distance_to(pred_edge)
    gt_pixels, pred_pixels = int(gt.sum()), int(pred.sum())
    errors = int(fp.sum() + fn.sum())
    recall = int(tp.sum()) / gt_pixels if gt_pixels else None
    result = dict(gt_pixels=gt_pixels, pred_pixels=pred_pixels, tp=int(tp.sum()),
                  fp=int(fp.sum()), fn=int(fn.sum()), error_pixels=errors,
                  gt_area_fraction=gt_pixels / gt.size,
                  size_bin=size_bin(gt_pixels / gt.size),
                  empty_prediction=int(pred_pixels == 0),
                  low_recall_lt_0_5=int(recall is not None and recall < .5))
    n_gt, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
    result['gt_components'] = n_gt - 1
    result['low_recall_gt_components'] = int(sum(
        np.count_nonzero(pred & (gt_labels == i)) / gt_stats[i, cv2.CC_STAT_AREA] < .5
        for i in range(1, n_gt)))
    n_pred, pred_labels, _, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    disjoint = [i for i in range(1, n_pred) if not np.any(gt[pred_labels == i])]
    result['disjoint_fp_components'] = len(disjoint)
    result['disjoint_fp_pixels'] = int(np.isin(pred_labels, disjoint).sum())
    for radius in RADII:
        near = gt_distance <= radius
        boundary_fp = int((fp & near).sum())
        boundary_fn = int((fn & near).sum())
        interior_fn = int((fn & ~near).sum())
        far_fp = int((fp & ~near).sum())
        assert boundary_fp + boundary_fn + interior_fn + far_fp == errors
        precision = float((gt_distance[pred_edge] <= radius).mean()) if pred_edge.any() else 0.
        sensitivity = float((pred_distance[gt_edge] <= radius).mean()) if gt_edge.any() else 0.
        bf1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.
        if not pred_edge.any() and not gt_edge.any():
            bf1 = 1.
        result.update({f'boundary_fp_{radius}px': boundary_fp,
                       f'boundary_fn_{radius}px': boundary_fn,
                       f'interior_fn_{radius}px': interior_fn,
                       f'far_fp_{radius}px': far_fp,
                       f'boundary_f1_{radius}px': bf1})
    # Both directed contours are pooled; undefined for one/both empty contours.
    distances = (np.concatenate((gt_distance[pred_edge], pred_distance[gt_edge]))
                 if pred_edge.any() and gt_edge.any() else None)
    result['surface_distance_mean_px'] = float(distances.mean()) if distances is not None else None
    result['surface_distance_p95_px'] = float(np.percentile(distances, 95)) if distances is not None else None
    return result


def summarize(rows):
    if not rows:
        return dict(images=0)
    total_error = sum(row['error_pixels'] for row in rows)
    result = dict(images=len(rows),
                  macro_metrics={key: float(np.mean([row[key] for row in rows]))
                                 for key in ('iou', 'dice', 'precision', 'recall')},
                  empty_predictions=sum(row['empty_prediction'] for row in rows),
                  low_recall_images=sum(row['low_recall_lt_0_5'] for row in rows),
                  low_recall_gt_components=sum(row['low_recall_gt_components'] for row in rows),
                  gt_components=sum(row['gt_components'] for row in rows),
                  disjoint_fp_components=sum(row['disjoint_fp_components'] for row in rows),
                  pixels={key: sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn', 'error_pixels')})
    result['error_partition'] = {}
    for radius in RADII:
        groups = {
            'boundary': [row[f'boundary_fp_{radius}px'] + row[f'boundary_fn_{radius}px'] for row in rows],
            'interior_fn': [row[f'interior_fn_{radius}px'] for row in rows],
            'far_fp': [row[f'far_fp_{radius}px'] for row in rows]}
        result['error_partition'][f'{radius}px'] = dict(
            mean_boundary_f1=float(np.mean([row[f'boundary_f1_{radius}px'] for row in rows])),
            pooled_error_fraction={key: sum(values) / total_error if total_error else 0.
                                   for key, values in groups.items()},
            # Perfect cases contribute zero to each per-image error fraction.
            macro_error_fraction={key: float(np.mean([value / row['error_pixels']
                if row['error_pixels'] else 0. for value, row in zip(values, rows)]))
                for key, values in groups.items()})
    return result


def case_panel(image, probability, target, prediction, row):
    """Four aligned panels; no resizing of masks relative to the image."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    gt_edge, pred_edge = boundary(target), boundary(prediction)
    contours = image.copy()
    contours[gt_edge] = (0, 255, 255)    # RGB cyan: GT
    contours[pred_edge] = (255, 0, 255)  # RGB magenta: prediction
    heat = cv2.cvtColor(cv2.applyColorMap(
        np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    truth = np.repeat((target.astype(np.uint8) * 255)[..., None], 3, axis=2)
    error = image.copy()
    fn, fp = target & ~prediction, prediction & ~target
    error[fn] = (.25 * error[fn] + .75 * np.array([255, 0, 0])).astype(np.uint8)
    error[fp] = (.25 * error[fp] + .75 * np.array([255, 255, 0])).astype(np.uint8)
    panels = []
    labels = ('Image: GT cyan / pred magenta', 'Probability (blue 0 / red 1)',
              'Ground truth', 'Errors: FN red / FP yellow')
    for panel, label in zip((contours, heat, truth, error), labels):
        panel = cv2.copyMakeBorder(panel, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(panel, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        panels.append(panel)
    grid = np.concatenate((np.concatenate(panels[:2], axis=1),
                           np.concatenate(panels[2:], axis=1)), axis=0)
    grid = cv2.copyMakeBorder(grid, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    title = f"{row['image_id']} IoU={row['iou']:.3f} recall={row['recall']:.3f} GT={100 * row['gt_area_fraction']:.2f}%"
    cv2.putText(grid, title, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
    return grid


def save_rgb(path, value):
    if not cv2.imwrite(str(path), cv2.cvtColor(value, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Cannot write image: {path}')


def evaluate(model, loader, evaluator, device, visual_dir):
    """No backward, optimizer, training augmentations, or checkpoint writes."""
    import torch
    from skimage import io

    rows = []
    model.eval()
    with torch.inference_mode():
        for images, masks, ids in loader:
            images = images.float().to(device)
            masks = masks.float().to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            probabilities = torch.sigmoid(model(images))
            predictions = (probabilities >= .5).float()
            metrics = evaluator.compute_batch_metrics(predictions, masks)
            for i, image_id in enumerate(ids):
                probability = probabilities[i, 0].cpu().numpy()
                pred = predictions[i, 0].cpu().numpy().astype(bool)
                gt = masks[i, 0].cpu().numpy().astype(bool)
                input_digest = hashlib.sha256(images[i].cpu().contiguous().numpy().tobytes()).hexdigest()
                mask_digest = hashlib.sha256(masks[i, 0].cpu().contiguous().numpy().tobytes()).hexdigest()
                row = dict(image_id=image_id, **{key: values[i] for key, values in metrics.items()},
                           **diagnose(pred, gt), input_sha256=input_digest, mask_sha256=mask_digest,
                           visual_file=f'cases/{len(rows):03d}_{Path(image_id).stem}.jpg')
                original = io.imread(str(Path(loader.dataset.path) / 'images' / image_id))[:, :, :3]
                original = cv2.resize(original, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                save_rgb(visual_dir / row['visual_file'], case_panel(original, probability, gt, pred, row))
                rows.append(row)
            print(f'  inferred {len(rows)}/{len(loader.dataset)} images', flush=True)
    return rows


def contact_sheets(rows, visual_dir):
    candidates = {
        'worst_iou': sorted(rows, key=lambda row: row['iou']),
        'small_low_recall': sorted((row for row in rows if row['gt_area_fraction'] < .05),
                                   key=lambda row: row['recall']),
        'far_background_fp': sorted((row for row in rows if row['far_fp_4px'] > 0),
                                    key=lambda row: row['far_fp_4px'], reverse=True),
        'boundary_error': sorted((row for row in rows if row['error_pixels']),
                                 key=lambda row: row['boundary_f1_4px'])}
    selection = []
    for name, ranked in candidates.items():
        selected = ranked[:12]
        if not selected:
            continue
        thumbs = []
        for rank, row in enumerate(selected, 1):
            image = cv2.imread(str(visual_dir / row['visual_file']))
            thumbs.append(cv2.resize(image, (512, 550)))
            selection.append(dict(category=name, rank=rank, image_id=row['image_id'],
                                  iou=row['iou'], recall=row['recall'], visual_file=row['visual_file']))
        blank = np.zeros_like(thumbs[0])
        thumbs += [blank] * ((-len(thumbs)) % 3)
        sheet = np.concatenate([np.concatenate(thumbs[i:i + 3], axis=1)
                                for i in range(0, len(thumbs), 3)], axis=0)
        if not cv2.imwrite(str(visual_dir / f'{name}.jpg'), sheet):
            raise OSError('Cannot write contact sheet')
    return selection


def preflight(args, source):
    ids = [line.strip() for line in (source / 'configs/splits/kvasir/val.txt').read_text().splitlines()
           if line.strip()]
    ids = [name if Path(name).suffix else name + '.jpg' for name in ids]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate validation IDs')
    data = []
    for name in ids:
        if Path(name).name != name:
            raise ValueError(f'Unexpected sample ID: {name}')
        image, mask = (args.dataset / part / name for part in ('images', 'masks'))
        data.append(dict(image_id=name, image_sha256=sha256(image), mask_sha256=sha256(mask)))
    runs = []
    for run_id in args.runs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
            raise ValueError(f'Expected a run directory name: {run_id}')
        record = args.output_root / 'experiments' / run_id / 'manifest.json'
        metadata = json.loads(record.read_text())
        if (metadata['status'] != 'completed' or metadata['source']['experiment'] != args.experiment
                or metadata.get('input_protocol') != 'paired_aug_v2'
                or metadata.get('numerical_protocol') != 'strict_cuda_v1'):
            raise ValueError(f'Not a completed strict-v2 {args.experiment}: {run_id}')
        names = ['networks/mslau_net.py', 'loader.py', 'kvasir_test.py', 'strict_runtime.py',
                 'configs/splits/kvasir/val.txt']
        if args.experiment != EXPERIMENT:
            names.append('networks/naf_discrete.py')
        for name in names:
            if sha256(source / name) != metadata['source']['files'][name]:
                raise ValueError(f'Evaluation source differs from training record: {run_id}: {name}')
        checkpoint = args.output_root / 'checkpoints' / run_id / 'kvasir_best_model.pth'
        actual = sha256(checkpoint)
        if actual != metadata['checkpoint_sha256']:
            raise ValueError(f'Checkpoint SHA256 mismatch: {checkpoint}')
        runs.append(dict(run_id=run_id, seed=metadata['seed'], checkpoint=str(checkpoint),
                         checkpoint_sha256=actual, manifest_sha256=sha256(record),
                         previous_metrics=metadata['evaluation_metrics']))
    if len({run['seed'] for run in runs}) != len(runs):
        raise ValueError('Pass one preselected checkpoint per seed; do not pool repeats as seeds')
    return data, runs


def cross_seed(rows_by_run):
    grouped = {}
    for run_id, rows in rows_by_run.items():
        for row in rows:
            grouped.setdefault(row['image_id'], []).append((run_id, row))
    result = []
    for image_id, items in grouped.items():
        rows = [item[1] for item in items]
        if (len(rows) != len(rows_by_run)
                or len({row['input_sha256'] for row in rows}) != 1
                or len({row['mask_sha256'] for row in rows}) != 1):
            raise ValueError(f'Cross-seed evaluation input mismatch: {image_id}')
        worst_run, worst = min(items, key=lambda item: item[1]['iou'])
        result.append(dict(image_id=image_id, size_bin=worst['size_bin'],
                           gt_area_fraction=worst['gt_area_fraction'],
                           mean_iou=float(np.mean([row['iou'] for row in rows])),
                           min_iou=min(row['iou'] for row in rows), max_iou=max(row['iou'] for row in rows),
                           runs=len(rows), runs_iou_lt_0_5=sum(row['iou'] < .5 for row in rows),
                           runs_low_recall=sum(row['low_recall_lt_0_5'] for row in rows),
                           mean_far_fp_4px=float(np.mean([row['far_fp_4px'] for row in rows])),
                           worst_run=worst_run, worst_visual_file=worst['visual_file'],
                           suspected_cause='', review_notes=''))
    return sorted(result, key=lambda row: row['mean_iou'])


def worker(args):
    import torch
    import albumentations
    import timm

    sys.path.insert(0, str(args.worker_source))
    import kvasir_test as evaluator
    from strict_runtime import configure_runtime, runtime_record

    configure_runtime()
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('Selected CUDA device is unavailable; no CPU fallback')
    data, runs = preflight(args, args.worker_source)
    records = args.output_root / 'experiments' / args.analysis_id
    visuals = args.output_root / 'visualization' / args.analysis_id
    if records.exists() or visuals.exists():
        raise FileExistsError('Analysis ID exists; use a new --analysis-id (nothing overwritten)')
    records.mkdir(parents=True)
    visuals.mkdir(parents=True)
    summary = dict(status='running', analysis_id=args.analysis_id, inference_only=True,
                   source=json.loads((args.worker_source / 'build_manifest.json').read_text()),
                   script_sha256=sha256(Path(__file__)), dataset=str(args.dataset),
                   split_sha256=sha256(args.worker_source / 'configs/splits/kvasir/val.txt'),
                   images=len(data), img_size=352, batch=8, num_workers=0, threshold=.5,
                   radii_pixels=RADII, area_bin_edges=[.01, .05, .20],
                   low_recall_threshold=.5, model_download=False,
                   python=sys.executable, versions=dict(torch=torch.__version__,
                   cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                   timm=timm.__version__, albumentations=albumentations.__version__,
                   numpy=np.__version__, opencv=cv2.__version__),
                   gpu=torch.cuda.get_device_name(0), runtime=runtime_record(), runs={})
    write_json(records / 'summary.json', summary)
    write_csv(records / 'dataset_files.csv', data)
    try:
        dataset = evaluator.binary_class(str(args.dataset), [row['image_id'] for row in data],
                                         evaluator.get_transform(352))
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False,
                                             num_workers=0, pin_memory=True)
        # Full saved state contains Hiera; pretrained=False avoids network access.
        torch.manual_seed(1234)
        model = evaluator.MSLAU_net(img_size=352, mla_channels=64, in_chans=3, num_classes=1,
                                    decoder_mode='p5_sam2unet', encoder_name='sam2_hiera_large')
        model.requires_grad_(False)
        model.to(device)
        rows_by_run = {}
        size_rows = []
        for run in runs:
            run_id = run['run_id']
            print(f"Analyzing seed {run['seed']}: {run['checkpoint']}", flush=True)
            state = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)
            model.load_state_dict(state, strict=True)
            del state
            run_visual = visuals / run_id
            (run_visual / 'cases').mkdir(parents=True)
            rows = evaluate(model, loader, evaluator, device, run_visual)
            metrics = summarize(rows)
            # The original evaluator logs rounded macro metrics, not raw training values.
            for current, previous in (('iou', 'IoU'), ('dice', 'dice')):
                if abs(metrics['macro_metrics'][current] - run['previous_metrics'][previous]) > .000051:
                    raise ValueError(f'Re-evaluation differs from recorded {previous}: {run_id}')
            rows_by_run[run_id] = rows
            run_records = records / run_id
            run_records.mkdir()
            write_csv(run_records / 'per_image.csv', rows)
            write_csv(run_records / 'review_candidates.csv', contact_sheets(rows, run_visual))
            by_size = {name: summarize([row for row in rows if row['size_bin'] == name]) for name in SIZE_BINS}
            for name, values in by_size.items():
                size_rows.append(dict(run_id=run_id, seed=run['seed'], size_bin=name,
                                      images=values['images'],
                                      mean_iou=values.get('macro_metrics', {}).get('iou'),
                                      mean_recall=values.get('macro_metrics', {}).get('recall'),
                                      low_recall_images=values.get('low_recall_images', 0)))
            summary['runs'][run_id] = dict(**run, **metrics, by_size=by_size,
                                           matches_recorded_evaluation=True)
            write_json(records / 'summary.json', summary)
        write_csv(records / 'size_groups.csv', size_rows)
        write_csv(records / 'cross_seed_review.csv', cross_seed(rows_by_run))
        summary['status'] = 'completed'
        summary['cross_seed_inputs_match'] = True
        summary['mean_seed_iou'] = float(np.mean([v['macro_metrics']['iou'] for v in summary['runs'].values()]))
        write_json(records / 'summary.json', summary)
        print(f'Completed inference only. Records: {records}\nImages: {visuals}', flush=True)
    except BaseException as exc:
        summary['status'] = 'failed'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        write_json(records / 'summary.json', summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=EXPERIMENTS, default=EXPERIMENT)
    parser.add_argument('--runs', nargs='+', required=True, help='completed strict-v2 run IDs, one per seed')
    parser.add_argument('--dataset', type=Path, default=Path('/data/dataset/zsy/mslau-net/Kvasir-SEG'))
    parser.add_argument('--output-root', type=Path,
                        default=Path(os.environ.get('P51_OUTPUT_ROOT', '/data/models/le/mslau-net')))
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--analysis-id')
    parser.add_argument('--worker-source', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.analysis_id is None:
        prefix = 'baseline_errors_v2_' if args.experiment == EXPERIMENT else 'exp09_errors_v2_'
        args.analysis_id = prefix + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.analysis_id) or args.gpu < 0:
        parser.error('Use a simple new analysis ID and a nonnegative physical GPU index')
    args.dataset = args.dataset.resolve()
    args.output_root = args.output_root.resolve()
    if args.worker_source:
        worker(args)
        return
    with tempfile.TemporaryDirectory(prefix='p51_error_analysis_') as temporary:
        source = Path(temporary) / 'source'
        build(args.experiment, source)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-source', str(source),
                   '--dataset', str(args.dataset), '--output-root', str(args.output_root),
                   '--analysis-id', args.analysis_id, '--gpu', str(args.gpu),
                   '--experiment', args.experiment, '--runs', *args.runs]
        subprocess.run(command, check=True, cwd=ROOT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                                CUBLAS_WORKSPACE_CONFIG=':4096:8', NO_ALBUMENTATIONS_UPDATE='1',
                                PYTHONDONTWRITEBYTECODE='1'))


if __name__ == '__main__':
    main()
