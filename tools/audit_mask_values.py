"""Read-only mask-value sensitivity check; never changes training/evaluation GT."""
import argparse
import datetime
import os
from pathlib import Path

import cv2
import numpy as np
from skimage import io, util

from analyze_baseline_errors import ROOT, sha256, write_csv, write_json
from build_experiment import verify_base


def compare_masks(raw):
    gray = util.img_as_float(raw)
    if gray.ndim != 2 or not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1:
        raise ValueError('Expected a finite grayscale mask in [0,1] after dtype conversion')
    masks = [cv2.resize((gray > threshold).astype(np.uint8), (352, 352),
                        interpolation=cv2.INTER_NEAREST) for threshold in (0., .5)]
    current, diagnostic = masks
    sizes = []
    for mask in masks:
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        sizes.append((len(areas), int((areas <= 4).sum()), int(areas[areas <= 4].sum())))
    union = int(np.logical_or(current, diagnostic).sum())
    row = dict(original_height=gray.shape[0], original_width=gray.shape[1],
               distinct_values=int(np.unique(gray).size),
               intermediate_fraction=float(((gray > 0) & (gray < 1)).mean()),
               low_gray_fraction=float(((gray > 0) & (gray <= .5)).mean()),
               current_gt_pixels=int(current.sum()), diagnostic_gt_pixels=int(diagnostic.sum()),
               added_pixels_gt0=int((current > diagnostic).sum()),
               diagnostic_mask_iou=float((np.logical_and(current, diagnostic).sum() + 1) / (union + 1)),
               current_components=sizes[0][0], diagnostic_components=sizes[1][0],
               current_tiny_components_le4px=sizes[0][1], diagnostic_tiny_components_le4px=sizes[1][1],
               current_tiny_pixels=sizes[0][2], diagnostic_tiny_pixels=sizes[1][2])
    return row, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/data/dataset/zsy/mslau-net/Kvasir-SEG'))
    parser.add_argument('--output-root', type=Path,
                        default=Path(os.environ.get('P51_OUTPUT_ROOT', '/data/models/le/mslau-net')))
    args = parser.parse_args()
    verify_base()
    split = ROOT / 'experiments/exp00_baseline/configs/splits/kvasir/val.txt'
    rows = []
    for line in split.read_text().splitlines():
        name = line.strip()
        if not name:
            continue
        name = name if Path(name).suffix else name + '.jpg'
        if Path(name).name != name:
            raise ValueError('Unexpected sample ID')
        path = args.dataset / 'masks' / name
        metrics, _ = compare_masks(io.imread(str(path), as_gray=True))
        rows.append(dict(image_id=name, mask_sha256=sha256(path), **metrics))
    run_id = 'mask_values_v2_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    output = args.output_root / 'experiments' / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'per_mask.csv', rows)
    summary = dict(status='completed', diagnostic_only=True, training_gt_unchanged=True,
                   dataset=str(args.dataset.resolve()), split_sha256=sha256(split), images=len(rows),
                   current_threshold='gray > 0', diagnostic_threshold='gray > 0.5 after dtype normalization',
                   resize='threshold first, nearest resize to 352; same order as original loader',
                   mean_diagnostic_mask_iou=float(np.mean([r['diagnostic_mask_iou'] for r in rows])),
                   min_diagnostic_mask_iou=min(r['diagnostic_mask_iou'] for r in rows),
                   median_current_components=float(np.median([r['current_components'] for r in rows])),
                   median_diagnostic_components=float(np.median([r['diagnostic_components'] for r in rows])),
                   total_added_pixels_gt0=sum(r['added_pixels_gt0'] for r in rows),
                   total_current_gt_pixels=sum(r['current_gt_pixels'] for r in rows),
                   note='Threshold 0.5 is a sensitivity probe, not a corrected annotation or an authorized protocol change.')
    write_json(output / 'summary.json', summary)
    print(f'Mask sensitivity check completed on CPU: {output}', flush=True)
    print(f"Mean mask IoU: {summary['mean_diagnostic_mask_iou']:.6f}; median component count: "
          f"{summary['median_current_components']} -> {summary['median_diagnostic_components']}")


if __name__ == '__main__':
    main()
