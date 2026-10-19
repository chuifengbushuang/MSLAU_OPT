"""Migrate nine historical matched runs, then delete only inventoried old outputs.

Default: print plan only. --apply performs the user-authorized migration/deletion.
Never scans or deletes code repositories, datasets, environments or zsy outputs.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

PLAN = Path(__file__).with_name('legacy_results_plan.json')
PYTHON = '/home/le/miniconda3/envs/mslau-p51/bin/python'
WEIGHTS = {'.pth', '.pt', '.ckpt', '.safetensors', '.onnx'}
IMAGES = {'.png', '.jpg', '.jpeg', '.svg'}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_scope(plan):
    root = Path(plan['source_root'])
    if root.resolve() != root or not root.is_dir():
        raise RuntimeError('Unexpected/missing source root')
    expected = plan['expected_old_files']
    for name in plan['remove_dirs'] + plan['remove_files']:
        if Path(name).name != name or name in {'MSLAU_OPT', 'MSLAU_OPT_original_p51', '.', '..'}:
            raise RuntimeError('Invalid deletion scope')
        path = root / name
        if path.is_symlink():
            raise RuntimeError(f'Refusing symlink: {path}')
        paths = path.rglob('*') if path.is_dir() else [path]
        for child in paths:
            if child.is_symlink():
                raise RuntimeError(f'Refusing symlink: {child}')
            if child.is_file() and (str(child) not in expected or child.stat().st_size != expected[str(child)]):
                raise RuntimeError(f'Old output changed since inventory. Stop before deletion: {child}')


def migrate(plan):
    output = Path(plan['output_root'])
    completed = []
    for item in plan['keep']:
        source = Path(item['source'])
        if source.parent.name not in plan['remove_dirs'] or not source.is_relative_to(Path(plan['source_root'])):
            raise RuntimeError(f'Invalid retained source: {source}')
        records = output / 'experiments' / item['id']
        checkpoint = output / 'checkpoints' / item['id'] / 'kvasir_best_model.pth'
        manifest = records / 'manifest.json'
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved['source'] != str(source) or saved['checkpoint_sha256'] != digest(Path(saved['checkpoint'])):
                raise RuntimeError(f'Existing migrated run does not match: {records}')
            if set(saved.get('checkpoint_files', {})) != set(item['checkpoint_names'].values()):
                raise RuntimeError('This run used the old single-checkpoint layout. Use restore_checkpoint_names.py; no old outputs deleted.')
            for name, expected in saved['checkpoint_files'].items():
                if digest(checkpoint.parent / name) != expected:
                    raise RuntimeError(f'Migrated checkpoint changed: {name}')
            for name, expected in saved.get('copied_records_sha256', {}).items():
                if digest(Path(name)) != expected:
                    raise RuntimeError(f'Migrated record changed: {name}')
            completed.append(saved)
            continue
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        records.mkdir(parents=True, exist_ok=True)
        checkpoint_files = {}
        for name in item['checkpoint_names'].values():
            if Path(name).name != name:
                raise RuntimeError('Invalid checkpoint filename')
            original = source / 'checkpoints' / name
            expected_bytes = plan['expected_old_files'].get(str(original))
            if not original.is_file() or original.stat().st_size != expected_bytes:
                raise RuntimeError(f'Missing/changed retained checkpoint: {original}')
            original_hash = digest(original)
            # CPU state_dict check, without constructing or training a model.
            keys = subprocess.check_output([PYTHON, '-c',
                'import sys,torch; s=torch.load(sys.argv[1],map_location="cpu",weights_only=True); '
                'assert isinstance(s,dict) and s and all(isinstance(v,torch.Tensor) for v in s.values()); print(len(s))',
                str(original)], text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='')).strip()
            if int(keys) != (1180 if item['variant'] == 'baseline' else 1200):
                raise RuntimeError(f'Unexpected model state: {original}, keys={keys}')
            target = checkpoint.parent / name
            shutil.copy2(original, target)
            if digest(target) != original_hash:
                raise RuntimeError(f'Copy verification failed: {target}')
            checkpoint_files[name] = original_hash
        copied_records = {}
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix not in WEIGHTS:
                base = output / 'visualization' / item['id'] if path.suffix.lower() in IMAGES else records
                target = base / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied_records[str(target)] = digest(path)
                if digest(target) != copied_records[str(target)]:
                    raise RuntimeError(f'Record copy verification failed: {target}')
        for path in source.parent.iterdir():
            if path.is_file() and path.suffix in {'.txt', '.json', '.patch', '.log', '.sh'}:
                target = records / 'batch_metadata' / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied_records[str(target)] = digest(path)
                if digest(target) != copied_records[str(target)]:
                    raise RuntimeError(f'Metadata copy verification failed: {target}')
        saved = dict(item, source=str(source), checkpoint=str(checkpoint),
                     checkpoint_sha256=checkpoint_files[checkpoint.name], state_dict_keys=int(keys),
                     checkpoint_files=checkpoint_files,
                     copied_records_sha256=copied_records,
                     status='historical_completed', comparison_group='old_mslau_naf_only')
        manifest.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + '\n')
        completed.append(saved)
        print('Preserved:', item['id'], flush=True)
    return completed


def apply_plan(plan):
    running = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'args='], text=True)
    if any('kvasir_train.py' in line or 'kvasir_test.py' in line for line in running.splitlines()):
        raise RuntimeError('A Kvasir training/evaluation process is running. No files deleted.')
    check_scope(plan)
    output = Path(plan['output_root'])
    required = sum(plan['expected_old_files'][str(Path(item['source']) / 'checkpoints' / name)]
                   for item in plan['keep'] for name in item['checkpoint_names'].values()
                   if not (output / 'experiments' / item['id'] / 'manifest.json').exists())
    if shutil.disk_usage(output).free < required + 10 * 1024 * 1024:
        raise RuntimeError(f'Insufficient free space for verified migration: need about {required / 2**30:.2f} GiB')
    completed = migrate(plan)
    assert len(completed) == 9
    records = Path(plan['output_root']) / 'experiments'
    # Retain a small summary of the discarded branches before deleting their outputs.
    with (records / 'discarded_history_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plan['historical_summary'][0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(plan['historical_summary'])
    (records / 'legacy_migration.json').write_text(json.dumps(completed, ensure_ascii=False, indent=2) + '\n')
    check_scope(plan)
    for item in completed:
        for name, expected in item['checkpoint_files'].items():
            if digest(Path(item['checkpoint']).parent / name) != expected:
                raise RuntimeError('Retained model changed; old outputs not deleted')
        for name, expected in item.get('copied_records_sha256', {}).items():
            if digest(Path(name)) != expected:
                raise RuntimeError('Retained record changed; old outputs not deleted')
    root = Path(plan['source_root'])
    for name in plan['remove_dirs']:
        if (root / name).exists():
            shutil.rmtree(root / name)
            print('Deleted old directory:', name, flush=True)
    for name in plan['remove_files']:
        (root / name).unlink(missing_ok=True)
    print('Done. Nine historical runs migrated; inventoried old outputs removed. No training started.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    print('KEEP: 9 historical runs, all three original checkpoints and records per run')
    for item in plan['keep']:
        print(' ', item['id'], '<-', item['source'])
    print('AFTER VERIFIED MIGRATION, DELETE old directories:', ', '.join(plan['remove_dirs']))
    print('DELETE old loose files:', ', '.join(plan['remove_files']))
    if not args.apply:
        print('PREVIEW ONLY. No files copied/deleted. Repositories and environments are excluded.')
        return
    if plan['source_root'] != '/home/le/WorkSpace/20261012_PolypSegmentation' or plan['output_root'] != '/data/models/le/mslau-net':
        raise RuntimeError('This one-time plan is restricted to the le server paths')
    apply_plan(plan)


if __name__ == '__main__':
    main()
