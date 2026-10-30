"""Materialize the exact PraNet archives and an overlay for the author's runner."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / 'work/polyper_official_20261030'
OUT = ROOT / 'analysis/20261030_polyper_official'
SOURCE = ROOT / 'work/paper_models_20261030/vendor/polyper'
FRAMEWORK = SOURCE / 'mmsegmentation'
BASE = FRAMEWORK / 'local_config/Polyper-AAAI2024/main/polyper_polypseg_224*224_80k.py'
ARCHIVES = {
    'TrainDataset.zip': ('1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb',
                         '30c99744df6770a4f514f8ccfa7404291ba968079ea3b6e3231ae74ad2b247cc'),
    'TestDataset.zip': ('1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao',
                        'b3be6da5b25679bed228dca52ae3d668dd89f329b2505c7b8961aa20dca4a90a'),
}
SIZES = {'Kvasir': 100, 'CVC-ClinicDB': 62, 'CVC-ColonDB': 380,
         'CVC-300': 60, 'ETIS-LaribPolypDB': 196}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def materialize(archive, image_prefix, mask_prefix, dest, count):
    images = sorted(n for n in archive.namelist()
                    if n.startswith(image_prefix + '/') and n.endswith('.png'))
    masks = {Path(n).name for n in archive.namelist()
             if n.startswith(mask_prefix + '/') and n.endswith('.png')}
    assert len(images) == count and masks == {Path(n).name for n in images}
    (dest / 'images').mkdir(parents=True, exist_ok=True)
    (dest / 'masks').mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in images:
        name = Path(entry).name
        raw_image = archive.read(entry)
        raw_mask = archive.read(mask_prefix + '/' + name)
        image = Image.open(io.BytesIO(raw_image))
        mask = Image.open(io.BytesIO(raw_mask))
        values = np.asarray(mask)
        assert image.size == mask.size and values.ndim == 2
        assert set(np.unique(values)) <= {0, 1, 255} and values.any()
        # PraNet masks are 1-bit PNG. OpenCV expands them to 0/255, while
        # MMSeg treats 255 as ignore. Preserve exact foreground support in
        # explicit uint8 class IDs 0/1; do not resample or threshold JPEGs.
        labels = (values > 0).astype(np.uint8)
        image_path, mask_path = dest / 'images' / name, dest / 'masks' / name
        image_path.write_bytes(raw_image)
        Image.fromarray(labels).save(mask_path)
        rows.append(dict(id=Path(name).stem, image_sha256=sha(image_path),
                         mask_sha256=sha(mask_path), width=image.width,
                         height=image.height, foreground_pixels=int(labels.sum()),
                         archive_mask_sha256=hashlib.sha256(raw_mask).hexdigest(),
                         archive_mask_mode=mask.mode))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archives = {}
    for name, (drive_id, expected) in ARCHIVES.items():
        p = WORK / 'downloads' / name
        assert sha(p) == expected, f'Archive checksum mismatch: {p}'
        archives[name] = dict(sha256=expected, bytes=p.stat().st_size,
                              url=f'https://drive.google.com/file/d/{drive_id}/view')
    data = WORK / 'data'
    with zipfile.ZipFile(WORK / 'downloads/TrainDataset.zip') as z:
        train = materialize(z, 'TrainDataset/image', 'TrainDataset/masks',
                            data / 'TrainDataset', 1450)
    kvasir_ids = {p.stem for p in (ROOT / 'datasets/Kvasir-SEG/images').iterdir() if p.is_file()}
    assert len(kvasir_ids) == 1000
    assert len({r['id'] for r in train} & kvasir_ids) == 900
    assert {r['id'] for r in train if not r['id'].isdigit()} <= kvasir_ids
    assert sum(r['id'].isdigit() for r in train) == 550
    tests = {}
    with zipfile.ZipFile(WORK / 'downloads/TestDataset.zip') as z:
        for name, count in SIZES.items():
            tests[name] = materialize(z, f'TestDataset/{name}/images',
                                     f'TestDataset/{name}/masks',
                                     data / 'TestDataset' / name, count)
    for name in ('Kvasir', 'CVC-ClinicDB'):
        assert not ({r['id'] for r in train} & {r['id'] for r in tests[name]})
        assert not ({r['image_sha256'] for r in train} &
                    {r['image_sha256'] for r in tests[name]})
    manifest = dict(archives=archives, training_counts=dict(Kvasir=900, ClinicDB=550),
                    train=train, tests=tests, mask_preparation='Exact PNG support -> uint8 class IDs 0/1; images byte-identical',
                    official_split_not_randomly_recreated=True)
    write(OUT / 'data_manifest.json', manifest)
    # Hooks collect records only. Network, loss, optimizer, sampler, transforms,
    # schedule, native prediction threshold and metric implementation inherit
    # the unmodified author source/config.
    pretrain = ROOT / 'work/paper_models_20261030/weights/swin_tiny.pth'
    assert sha(pretrain) == '3f89aee6e380026163f9f709ed14fc1726e0f8e2578d1f2d12bb465bbc8ae237'
    common = f'''_base_ = {str(BASE)!r}
custom_imports = dict(imports=['polyper_official_hooks'], allow_failed_imports=False)
randomness = dict(seed=1234, deterministic=False)
model = dict(backbone=dict(init_cfg=dict(checkpoint={str(pretrain)!r})))
train_dataloader = dict(dataset=dict(dataset=dict(data_root='', data_prefix=dict(
    img_path={str(data / 'TrainDataset/images')!r}, seg_map_path={str(data / 'TrainDataset/masks')!r}))))
val_dataloader = dict(dataset=dict(data_root='', data_prefix=dict(
    img_path={str(data / 'TestDataset/Kvasir/images')!r}, seg_map_path={str(data / 'TestDataset/Kvasir/masks')!r})))
test_dataloader = dict(dataset=dict(data_root='', data_prefix=dict(
    img_path={str(data / 'TestDataset/Kvasir/images')!r}, seg_map_path={str(data / 'TestDataset/Kvasir/masks')!r})))
val_evaluator = dict(iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(iou_metrics=['mIoU', 'mDice'])
custom_hooks = [dict(type='OfficialProtocolRecordHook', formal=True)]
'''
    config = WORK / 'official_80k.py'
    config.write_text(common)
    # Independent, disposable preflight: two real batches and two native-size
    # test images. Resume to a third update verifies full optimizer loading.
    ids = sorted(r['id'] for r in tests['Kvasir'])[:2]
    (WORK / 'smoke_ids.txt').write_text('\n'.join(ids) + '\n')
    smoke = f'''_base_ = {str(config)!r}
train_cfg = dict(max_iters=2, val_interval=2)
val_dataloader = dict(dataset=dict(ann_file={str(WORK / 'smoke_ids.txt')!r}))
test_dataloader = dict(dataset=dict(ann_file={str(WORK / 'smoke_ids.txt')!r}))
default_hooks = dict(checkpoint=dict(interval=1), logger=dict(interval=1))
custom_hooks = [dict(type='OfficialProtocolRecordHook', formal=False)]
'''
    (WORK / 'smoke.py').write_text(smoke)
    (WORK / 'smoke_resume.py').write_text(
        f'_base_ = {str(WORK / "smoke.py")!r}\ntrain_cfg = dict(max_iters=3, val_interval=3)\n')
    files = {str(p.relative_to(SOURCE)): sha(p) for p in sorted(SOURCE.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    write(OUT / 'source_manifest.json', dict(upstream=json.loads((SOURCE / 'upstream_manifest.json').read_text()),
          files=files, author_files_modified=False, official_base_config=str(BASE),
          pretrained_sha256=sha(pretrain), overlay_sha256=sha(config),
          added_sources={p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}))
    print(json.dumps(dict(train=1450, tests=SIZES, config=str(config),
                          data_manifest_sha256=sha(OUT / 'data_manifest.json'))))


if __name__ == '__main__':
    main()
