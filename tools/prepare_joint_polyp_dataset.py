"""Keep CVC splits and Kvasir validation; hold out 80 old Kvasir train IDs."""
import argparse
import csv
import hashlib
import io
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
KVASIR_SPLITS = ROOT / 'experiments/exp00_baseline/configs/splits/kvasir'
CVC_SPLITS = ROOT / 'configs/splits/cvc_clinicdb_sequence_v1'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ids(path):
    names = path.read_text().splitlines()
    if not names or len(names) != len(set(names)) or any(
            name != name.strip() or Path(name).name != name or name in ('.', '..') for name in names):
        raise ValueError(f'Invalid/repeated sample IDs: {path}')
    return names


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def prepare(view, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refusing to overwrite a frozen joint split')
    cvc_manifest = json.loads((CVC_SPLITS / 'manifest.json').read_text())
    for name, digest in cvc_manifest['files_sha256'].items():
        if sha256(CVC_SPLITS / name) != digest:
            raise ValueError(f'Frozen CVC split changed: {name}')
    with (CVC_SPLITS / 'samples.csv').open(newline='') as stream:
        cvc_samples = {row['sample_id']: row for row in csv.DictReader(stream)}

    paths = [KVASIR_SPLITS / f'{phase}.txt' for phase in ('train', 'val', 'test')]
    paths += [CVC_SPLITS / name for name in ('train.txt', 'val.txt', 'test.txt', 'manifest.json', 'samples.csv')]
    original_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    kvasir = {phase: load_ids(KVASIR_SPLITS / f'{phase}.txt') for phase in ('train', 'val', 'test')}
    if kvasir['val'] != kvasir['test']:
        raise ValueError('Expected the existing mirrored Kvasir val/test protocol; inspect before changing it')
    cvc = {phase: load_ids(CVC_SPLITS / f'{phase}.txt') for phase in ('train', 'val', 'test')}
    if [len(kvasir[name]) for name in ('train', 'val')] != [880, 120]:
        raise ValueError('Kvasir split counts changed')
    if [len(cvc[name]) for name in ('train', 'val', 'test')] != [453, 74, 85]:
        raise ValueError('CVC split counts changed')
    held_out = set(random.Random(20261029).sample(kvasir['train'], 80))
    kvasir['test'] = [name for name in kvasir['train'] if name in held_out]
    kvasir['train'] = [name for name in kvasir['train'] if name not in held_out]

    rows, decoded_hashes = [], {}
    for phase in ('train', 'val', 'test'):
        for dataset, split in (('Kvasir-SEG', kvasir), ('CVC-ClinicDB', cvc)):
            for name in split[phase]:
                image, mask = (WORKSPACE / 'datasets' / dataset / sub / name for sub in ('images', 'masks'))
                if not image.is_file() or not mask.is_file():
                    raise FileNotFoundError(f'Missing image/mask: {dataset}/{name}')
                with Image.open(image) as im:
                    pixels = np.asarray(im.convert('RGB'))
                    image_size = im.size
                with Image.open(mask) as im:
                    im.load()
                    if im.size != image_size:
                        raise ValueError(f'Image/mask size mismatch: {dataset}/{name}')
                image_digest, mask_digest = sha256(image), sha256(mask)
                if dataset == 'CVC-ClinicDB':
                    prior = cvc_samples[name]
                    if image_digest != prior['image_sha256'] or mask_digest != prior['mask_sha256']:
                        raise ValueError(f'CVC data changed since its original audit: {name}')
                    if prior['split'] != phase:
                        raise ValueError(f'CVC membership mismatch: {name}')
                pixel_digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
                sample_id = f'{dataset}/{name}'
                decoded_hashes.setdefault(pixel_digest, []).append((phase, sample_id))
                rows.append(dict(sample_id=sample_id, dataset=dataset, original_id=name, split=phase,
                    sequence_id=cvc_samples[name]['sequence_id'] if dataset == 'CVC-ClinicDB' else '',
                    image_path=f'datasets/{dataset}/images/{name}', mask_path=f'datasets/{dataset}/masks/{name}',
                    image_sha256=image_digest, mask_sha256=mask_digest, decoded_rgb_sha256=pixel_digest))
    ids = [row['sample_id'] for row in rows]
    if len(ids) != 1612 or len(set(ids)) != len(ids):
        raise ValueError('Joint split has missing/overlapping IDs')
    if set(kvasir['train']) & set(kvasir['val']):
        raise ValueError('Existing Kvasir train/val overlap')
    sequence_sets = {phase: {row['sequence_id'] for row in rows if row['dataset'] == 'CVC-ClinicDB'
                             and row['split'] == phase} for phase in ('train', 'val', 'test')}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if sequence_sets[left] & sequence_sets[right]:
            raise ValueError('CVC sequences cross split boundaries')
    duplicates = [group for group in decoded_hashes.values() if len(group) > 1]
    if any(len({phase for phase, _ in group}) > 1 for group in duplicates):
        raise ValueError(f'Exact decoded duplicate crosses splits; original data needs inspection: {duplicates}')
    if any(sha256(ROOT / name) != digest for name, digest in original_hashes.items()):
        raise ValueError('Source splits changed during preparation')

    # Four directory links expose the original reader's images/<ID>, masks/<ID> API.
    # Relative link targets keep the view relocatable with the workspace.
    view.mkdir(parents=True, exist_ok=True)
    links = {}
    for sub in ('images', 'masks'):
        (view / sub).mkdir(exist_ok=True)
        for dataset in ('Kvasir-SEG', 'CVC-ClinicDB'):
            link = view / sub / dataset
            target = os.path.relpath(WORKSPACE / 'datasets' / dataset / sub, link.parent)
            if link.is_symlink():
                if link.resolve() != (WORKSPACE / 'datasets' / dataset / sub).resolve():
                    raise ValueError(f'Unexpected existing link: {link}')
            elif link.exists():
                raise FileExistsError(f'Expected a directory symlink: {link}')
            else:
                link.symlink_to(target, target_is_directory=True)
            links[str(link.relative_to(view))] = target
    files = {}
    for phase in ('train', 'val', 'test'):
        selected = [row for row in rows if row['split'] == phase]
        files[f'{phase}.txt'] = ''.join(row['sample_id'] + '\n' for row in selected)
    files['samples.csv'] = csv_text(rows)
    manifest = dict(protocol='kvasir_clinicdb_holdout_v1',
        roots_relative_to_workspace=dict(dataset_view=str(view.relative_to(WORKSPACE)),
                                         split_directory=str(output.relative_to(WORKSPACE))),
        data_storage='Four relative directory symlinks; no images/masks copied, renamed, resized or rewritten.',
        links=links, counts=dict(train=1253, val=194, test=165),
        counts_by_source=dict(train={'Kvasir-SEG': 800, 'CVC-ClinicDB': 453},
                              val={'Kvasir-SEG': 120, 'CVC-ClinicDB': 74},
                              test={'Kvasir-SEG': 80, 'CVC-ClinicDB': 85}),
        kvasir_holdout=dict(seed=20261029, count=80,
            method='random.Random(seed).sample(original_train_list,80), one draw; no model scores',
            limitation='Held out only for fresh runs. Historical Kvasir models trained on these images; no patient grouping available.'),
        ordering='Original relative list order within each subset; Kvasir then CVC; CVC membership unchanged.',
        sample_id='dataset/original_filename; input augmentation and frequency-selection keys use this complete ID.',
        old_kvasir_test='Original mirrored test.txt remains unchanged, excluded from joint independent test.',
        validation_selection='Arithmetic mean of all 194 per-image IoUs, not equal weighting of dataset means.',
        test_limitations='165-image internal holdout; Kvasir80+CVC85. CVC has 4 local sequence IDs; no external/patient-level generalization claim.',
        group_limitations=cvc_manifest['grouping_limitation'],
        original_split_files_sha256=original_hashes,
        files_sha256={name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()},
        generator_sha256=sha256(Path(__file__)), exact_decoded_duplicate_groups=duplicates,
        checks=dict(cvc_membership_and_kvasir_validation_preserved=True, all_1612_pairs_decoded=True,
                    source_split_bytes_unchanged=True, joint_ids_disjoint=True,
                    cvc_sequences_disjoint=True, no_exact_decoded_duplicates_across_splits=True))
    output.mkdir(parents=True, exist_ok=False)
    for name, body in files.items():
        (output / name).write_text(body, encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', type=Path, default=WORKSPACE / 'datasets/Polyp-Mixed-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/splits/kvasir_clinicdb_holdout_v1')
    args = parser.parse_args()
    manifest = prepare(args.view.absolute(), args.output.absolute())
    print(json.dumps(dict(view=str(args.view), splits=str(args.output), counts=manifest['counts'],
                          exact_decoded_duplicate_groups=manifest['exact_decoded_duplicate_groups']), indent=2))


if __name__ == '__main__':
    main()
