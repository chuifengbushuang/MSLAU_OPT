"""Audit local ClinicDB PNGs and freeze a sequence-disjoint split; no training."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEED = 20261029


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rgb(path):
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.size != (384, 288):
            raise ValueError(f'Unexpected PNG format: {path}: {image.mode}/{image.size}')
        image.load()
        return np.asarray(image).copy()


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def prepare(dataset, output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite frozen split: {output}')
    metadata = dataset / 'metadata.csv'
    with metadata.open(newline='') as stream:
        source_rows = list(csv.DictReader(stream))
    frames = [int(row['frame_id']) for row in source_rows]
    if len(frames) != 612 or set(frames) != set(range(1, 613)):
        raise ValueError('Expected each frame 1..612 exactly once in metadata')
    groups = sorted({int(row['sequence_id']) for row in source_rows})
    if groups != list(range(1, 30)):
        raise ValueError('Expected the 29 local metadata sequence IDs')
    for sub in ('images', 'masks', 'PNG/Original', 'PNG/Ground Truth'):
        actual = {path.name for path in (dataset / sub).iterdir() if path.is_file()}
        if actual != {f'{frame}.png' for frame in frames}:
            raise ValueError(f'Unexpected/missing PNG files in {sub}')

    # One fixed shuffle; no search over allocations, foreground balance or scores.
    random.Random(SPLIT_SEED).shuffle(groups)
    assignment = {g: phase for phase, ids in (
        ('train', groups[:21]), ('val', groups[21:25]), ('test', groups[25:])) for g in ids}
    samples, pixel_hashes = [], {}
    for row in sorted(source_rows, key=lambda item: int(item['frame_id'])):
        frame, sequence = int(row['frame_id']), int(row['sequence_id'])
        name = f'{frame}.png'
        if (row['png_image_path'] != f'PNG/Original/{name}' or
                row['png_mask_path'] != f'PNG/Ground Truth/{name}'):
            raise ValueError(f'Metadata PNG path mismatch: {frame}')
        image, mask = (read_rgb(dataset / sub / name) for sub in ('images', 'masks'))
        for original, sub in ((image, 'PNG/Original'), (mask, 'PNG/Ground Truth')):
            if not np.array_equal(original, read_rgb(dataset / sub / name)):
                raise ValueError(f'PNG copy pixel mismatch: {sub}/{name}')
        if not (np.array_equal(mask[:, :, 0], mask[:, :, 1]) and
                np.array_equal(mask[:, :, 1], mask[:, :, 2])):
            raise ValueError(f'Mask channels differ: {name}')
        gray = mask[:, :, 0]
        current, diagnostic = gray > 0, gray > 127
        if not current.any() or current.all() or not diagnostic.any():
            raise ValueError(f'Empty/full/unexpected mask: {name}')
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
        pixel_hashes.setdefault(pixel_hash, []).append(name)
        samples.append(dict(
            sample_id=name, frame_id=frame, sequence_id=sequence, split=assignment[sequence],
            image_path=f'images/{name}', mask_path=f'masks/{name}', width=384, height=288,
            image_sha256=sha256(dataset / 'images' / name),
            mask_sha256=sha256(dataset / 'masks' / name), image_rgb_sha256=pixel_hash,
            foreground_fraction_gt0=float(current.mean()),
            foreground_fraction_gt127_diagnostic=float(diagnostic.mean()),
            intermediate_pixels=int(((gray > 0) & (gray < 255)).sum()),
            low_gray_pixels=int((current & ~diagnostic).sum()),
            threshold_mask_iou_diagnostic=float(diagnostic.sum() / current.sum())))
    duplicates = [ids for ids in pixel_hashes.values() if len(ids) > 1]
    if duplicates:
        raise ValueError(f'Duplicate decoded images require group reconciliation: {duplicates}')

    summaries, sequence_rows = {}, []
    for phase in ('train', 'val', 'test'):
        selected = [row for row in samples if row['split'] == phase]
        sequence_ids = sorted(g for g, part in assignment.items() if part == phase)
        summaries[phase] = dict(images=len(selected), sequences=len(sequence_ids),
            sequence_ids=sequence_ids,
            small_masks_lt5pct_gt0=sum(row['foreground_fraction_gt0'] < .05 for row in selected),
            mean_foreground_fraction_gt0=float(np.mean([row['foreground_fraction_gt0'] for row in selected])))
    for sequence in sorted(assignment):
        ids = [row['frame_id'] for row in samples if row['sequence_id'] == sequence]
        sequence_rows.append(dict(sequence_id=sequence, split=assignment[sequence],
                                  images=len(ids), first_frame=min(ids), last_frame=max(ids)))
    files = {f'{phase}.txt': ''.join(row['sample_id'] + '\n' for row in samples if row['split'] == phase)
             for phase in ('train', 'val', 'test')}
    files['samples.csv'] = csv_text(samples)
    files['sequences.csv'] = csv_text(sequence_rows)
    manifest = dict(schema_version=1, dataset='CVC-ClinicDB', protocol='clinicdb_sequence_v1',
        dataset_root=str(dataset.resolve()), metadata_sha256=sha256(metadata),
        generator_sha256=sha256(Path(__file__)), split_seed=SPLIT_SEED,
        method='random.Random(seed).shuffle(sorted(sequence_ids)); first 21 train, next 4 val, final 4 test',
        split_selection='single predeclared draw; no reroll/foreground balancing/model scores',
        grouping_source='local metadata.csv sequence_id; 29 groups; provenance not independently verified',
        grouping_limitation='Literature often reports 31 sequences/23 patients. No patient IDs locally; patient isolation is not established.',
        summaries=summaries, image_count=612, image_size_wh=[384, 288],
        storage='images/*.png and masks/*.png; pixel-identical to PNG/Original and PNG/Ground Truth',
        unused_storage='TIF is not used or validated; metadata tif_* paths say TIFF/, local folder is TIF/.',
        checks=dict(all_pngs_decoded=True, png_copies_pixel_identical=True,
                    image_ids_disjoint=True, sequence_ids_disjoint=True, coverage_612=True,
                    exact_decoded_image_duplicates=0, mask_channels_identical=True),
        mask_audit=dict(intermediate_value_masks=sum(row['intermediate_pixels'] > 0 for row in samples),
            low_gray_masks=sum(row['low_gray_pixels'] > 0 for row in samples),
            total_low_gray_pixels=sum(row['low_gray_pixels'] for row in samples),
            mean_gt0_vs_gt127_iou=float(np.mean([row['threshold_mask_iou_diagnostic'] for row in samples])),
            min_gt0_vs_gt127_iou=min(row['threshold_mask_iou_diagnostic'] for row in samples),
            training_threshold='original loader: skimage as_gray then >0, before nearest-neighbor resize',
            diagnostic_threshold='uint8 >127 only for annotation sensitivity; no files changed'),
        files_sha256={name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()})
    output.mkdir(parents=True, exist_ok=False)
    for name, body in files.items():
        (output / name).write_text(body, encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT.parent / 'datasets/CVC-ClinicDB')
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/splits/cvc_clinicdb_sequence_v1')
    args = parser.parse_args()
    manifest = prepare(args.dataset.resolve(), args.output.resolve())
    print(json.dumps(dict(output=str(args.output.resolve()), summaries=manifest['summaries'],
                         mask_audit=manifest['mask_audit']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
