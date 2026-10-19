"""Restore original names after the previous single-checkpoint migration.

Preview by default. Never creates a best-loss model from a best-IoU model.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from organize_legacy_results import PLAN, digest


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prepare(plan):
    output = Path(plan['output_root'])
    tasks = []
    for item in plan['keep']:
        directory = output / 'checkpoints' / item['id']
        record = output / 'experiments' / item['id'] / 'manifest.json'
        saved = json.loads(record.read_text())
        expected = saved['checkpoint_sha256']
        recorded = Path(saved['checkpoint'])
        if recorded.parent != directory or directory.is_symlink():
            raise RuntimeError(f'Unexpected checkpoint location: {recorded}')
        candidates = [recorded, directory / 'kvasir_best_model.pth', directory / 'best_iou.pth']
        primary = next((path for path in candidates if path.is_file()), None)
        if primary is None or primary.is_symlink() or digest(primary) != expected:
            raise RuntimeError(f'Retained best-IoU file missing/changed: {directory}')
        names = item['checkpoint_names']
        for name in names.values():
            if Path(name).name != name:
                raise RuntimeError('Invalid historical checkpoint filename')
        for key in ['fixed', 'best_iou']:
            target = directory / names[key]
            if target.exists() and (target.is_symlink() or digest(target) != saved.get('checkpoint_files', {}).get(target.name, expected)):
                raise RuntimeError(f'Existing file conflicts with retained model: {target}')
        loss = directory / names['best_loss']
        original_loss = Path(item['source']) / 'checkpoints' / names['best_loss']
        if original_loss.is_symlink() or loss.is_symlink():
            raise RuntimeError('Refusing checkpoint symlink')
        original_loss_exists = original_loss.is_file()
        if original_loss_exists and original_loss.stat().st_size != plan['expected_old_files'][str(original_loss)]:
            raise RuntimeError(f'Original best-loss file size changed: {original_loss}')
        loss_expected = digest(original_loss) if original_loss_exists else saved.get('checkpoint_files', {}).get(loss.name)
        if loss.exists() and (loss_expected is None or digest(loss) != loss_expected):
            raise RuntimeError(f'Existing best-loss file has no matching provenance: {loss}')
        tasks.append(dict(item=item, directory=directory, record=record, saved=saved,
                          primary=primary, original_loss=original_loss, loss=loss,
                          loss_expected=loss_expected, loss_available=loss.is_file() or original_loss_exists))
    return tasks


def restore(plan, apply=False):
    # Validate every retained model/name before modifying any run.
    tasks = prepare(plan)
    completed = []
    for task in tasks:
        item, directory, saved = task['item'], task['directory'], task['saved']
        names = item['checkpoint_names']
        print(item['id'], '->', names['fixed'], '+', names['best_iou'])
        print('  best-loss:', names['best_loss'], 'available' if task['loss_available'] else 'MISSING; cannot reconstruct from best-IoU')
        if not apply:
            continue
        fixed = directory / names['fixed']
        primary = task['primary']
        if not fixed.exists():
            if primary.name == 'best_iou.pth':
                primary.rename(fixed)
            else:
                os.link(primary, fixed)
        named_iou = directory / names['best_iou']
        if not named_iou.exists():
            # Two original IoU names refer to the same retained model, with no
            # additional large-file copy. Neither is labeled as best-loss.
            os.link(fixed, named_iou)
        if not task['loss'].exists() and task['original_loss'].is_file():
            shutil.copy2(task['original_loss'], task['loss'])
            if digest(task['loss']) != task['loss_expected']:
                raise RuntimeError('Best-loss copy verification failed')
        existing = {name: digest(directory / name) for name in names.values() if (directory / name).is_file()}
        saved.update(checkpoint=str(fixed), checkpoint_files=existing,
                     missing_checkpoints=[name for name in names.values() if name not in existing],
                     checkpoint_layout_note='Original names restored; an absent best-loss state is not reconstructed. Newly created IoU aliases use hard links.')
        write_json(task['record'], saved)
        completed.append(saved)
    if apply:
        write_json(Path(plan['output_root']) / 'experiments/legacy_migration.json', completed)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    tasks = restore(json.loads(PLAN.read_text()), args.apply)
    print('Applied.' if args.apply else 'PREVIEW ONLY; use --apply to restore names.')
    print('Runs with unavailable original best-loss:', sum(not task['loss_available'] for task in tasks))
    print('No training started. No old checkpoint directories deleted.')


if __name__ == '__main__':
    main()
