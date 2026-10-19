"""Materialize frozen baseline + one experiment's patch and added files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/exp00_baseline'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in {'.', '..', '.git'} for p in path.parts) or '\\' in name:
        raise ValueError(f'Unsafe experiment path: {name}')
    return path


def verify_base():
    source = json.loads((BASE / 'source_manifest.json').read_text())
    for name, expected in {**source['files'], **source['controls']}.items():
        path = BASE / safe_relative(name)
        if path.is_symlink() or digest(path) != expected:
            raise RuntimeError(f'Frozen baseline changed: {name}')
    return source


def build(experiment, output):
    if not re.fullmatch(r'exp\d{2,}_[a-zA-Z0-9_]+', experiment):
        raise ValueError('Use an experiment directory name such as exp00_baseline')
    definition = ROOT / 'experiments' / experiment
    if definition.is_symlink():
        raise ValueError('Experiment definition must not be a symlink')
    source = verify_base()
    config_path = definition / 'config.json'
    if config_path.is_symlink():
        raise ValueError('Config must not be a symlink')
    config = json.loads(config_path.read_text())
    if config['id'] != experiment or config['base_commit'] != source['source_commit']:
        raise ValueError('Experiment id/base_commit does not match the frozen baseline')
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists; nothing overwritten: {output}')
    if output.is_relative_to(BASE) or output.is_relative_to(definition):
        raise ValueError('Build output must be outside experiment definitions')
    patch = definition / 'changes.patch'
    added = definition / 'added'
    original_names = set(source['files']) | {'run.py'}
    if patch.is_symlink() or added.is_symlink():
        raise ValueError('Patch/added directory must not be a symlink')
    patch_hash = digest(patch) if patch.is_file() else None
    added_hashes = {}
    if added.is_dir():
        for path in added.rglob('*'):
            if path.is_symlink():
                raise ValueError(f'Added symlink is not allowed: {path}')
            if path.is_file():
                name = str(safe_relative(path.relative_to(added).as_posix()))
                if name in original_names or name in {'config.json', 'source_manifest.json', 'build_manifest.json', 'README.md'}:
                    raise ValueError(f'Existing/control file must not be overlaid through added/: {name}')
                if path.suffix in {'.pth', '.pt', '.ckpt', '.safetensors', '.onnx'}:
                    raise ValueError('Weights do not belong in experiment source')
                added_hashes[name] = digest(path)
    with tempfile.TemporaryDirectory(prefix='p51_source_') as temporary:
        stage = Path(temporary)
        for name in original_names | {'source_manifest.json'}:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BASE / name, target)
        if patch.is_file() and patch.stat().st_size:
            # stage is outside the repository; git apply cannot skip paths because
            # of the caller's repository prefix, or modify the frozen baseline.
            stats = subprocess.check_output(['git', 'apply', '--numstat', str(patch)], cwd=stage, text=True)
            if not stats.strip():
                raise ValueError('Patch contains no file changes')
            for line in stats.splitlines():
                insertions, deletions, name = line.split('\t')
                if '-' in {insertions, deletions} or str(safe_relative(name)) not in original_names:
                    raise ValueError(f'Patch must modify original text files; new files go in added/: {name}')
            subprocess.run(['git', 'apply', '--check', str(patch)], cwd=stage, check=True)
            subprocess.run(['git', 'apply', str(patch)], cwd=stage, check=True)
        for name in added_hashes:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(added / name, target)
        shutil.copy2(config_path, stage / 'config.json')
        readme = definition / 'README.md'
        if readme.is_file() and not readme.is_symlink():
            shutil.copy2(readme, stage / 'README.md')
        else:
            (stage / 'README.md').write_text(f'# {experiment}\nGenerated standalone source. Run with the desired Python environment: `python run.py --seed 1234 --gpu 0 --run`.\n')
        if not (stage / 'run.py').is_file():
            raise ValueError('Experiment must retain the standalone launcher')
        for path in stage.rglob('*'):
            if path.is_symlink():
                raise ValueError('Patched source must not contain symlinks')
        files = {p.relative_to(stage).as_posix(): digest(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
        manifest = dict(experiment=experiment, source_commit=source['source_commit'],
                        repository_commit=revision, worktree_dirty=dirty,
                        patch_sha256=patch_hash, config_sha256=digest(config_path),
                        added_files=added_hashes, files=files)
        (stage / 'build_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        output.mkdir(parents=True, exist_ok=False)
        try:
            shutil.copytree(stage, output, dirs_exist_ok=True)
        except BaseException:
            shutil.rmtree(output)
            raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    parser.add_argument('--output', type=Path, required=True, help='new directory for runtime source or standalone export')
    args = parser.parse_args()
    manifest = build(args.experiment, args.output)
    print(f"Built {manifest['experiment']}: {args.output}, {len(manifest['files'])} files. No training started.")


if __name__ == '__main__':
    main()
