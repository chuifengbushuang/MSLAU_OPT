"""Record Kvasir runs without changing model, loss, RNG or checkpoint semantics."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import uuid


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def capture(command, cwd):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'{shlex.join(command)}\n{result.stdout}')
    return result.stdout


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='include variant, seed and repeat')
    parser.add_argument('--kind', choices=['train', 'evaluate'], default='train')
    parser.add_argument('--parent-run', help='training run_id for evaluation provenance')
    parser.add_argument('--root', type=Path, default=Path('/data/models/le/mslau-net'))
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--gpu', required=True, help='one physical GPU index, or cpu')
    parser.add_argument('--dry-run', action='store_true', help='print plan; no files or child processes')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name):
        parser.error('--name must use letters, digits, underscore or hyphen')
    if args.gpu != 'cpu' and not args.gpu.isdigit():
        parser.error('--gpu must name one physical index or cpu')
    native = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    managed = {'--output_dir', '--log_dir', '--debug_dir'}
    if any(arg.split('=', 1)[0] in managed for arg in native):
        parser.error('output_dir/log_dir/debug_dir are managed by this launcher')
    repo = args.repo.resolve()
    run_id = f"{args.name}_{dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    paths = {key: str(args.root.resolve() / key / run_id)
             for key in ['checkpoints', 'experiments', 'visualization']}
    script = 'kvasir_train.py' if args.kind == 'train' else 'kvasir_test.py'
    command = [args.python, '-u', str(repo / script), *native]
    if args.kind == 'train':
        command += ['--output_dir', paths['checkpoints'], '--log_dir', paths['experiments'] + '/logs']
    else:
        command += ['--debug_dir', paths['visualization']]
    record = dict(run_id=run_id, kind=args.kind, parent_run=args.parent_run,
                  paths=paths, cwd=str(repo), command=command, gpu=args.gpu,
                  status='planned', started_at=dt.datetime.now().astimezone().isoformat())
    if args.dry_run:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    if not (repo / script).is_file():
        parser.error(f'missing entry point: {repo / script}')
    # Reserve a unique records directory. Never overwrite an existing run.
    records = Path(paths['experiments'])
    records.mkdir(parents=True, exist_ok=False)
    manifest = records / 'manifest.json'
    write_json(manifest, record)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if args.gpu == 'cpu' else args.gpu,
               NO_ALBUMENTATIONS_UPDATE='1', PYTHONUNBUFFERED='1')
    child = None
    old_handlers = {}
    try:
        for key in ['checkpoints', 'visualization']:
            Path(paths[key]).mkdir(parents=True, exist_ok=False)
        (records / 'command.txt').write_text(
            f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])} " + shlex.join(command) + '\n')
        for label, cmd in [('commit.txt', ['git', 'rev-parse', 'HEAD']),
                           ('status.txt', ['git', 'status', '--short']),
                           ('code_diff.patch', ['git', 'diff', 'HEAD', '--binary'])]:
            (records / label).write_text(capture(cmd, repo))
        # Include untracked source too; a commit+diff alone would omit it.
        files = capture(['git', 'ls-files', '-co', '--exclude-standard'], repo).splitlines()
        import zipfile
        with zipfile.ZipFile(records / 'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(set(files)):
                path = repo / name
                if path.is_file() and not path.is_symlink() and path.suffix not in {'.pth', '.pt', '.ckpt', '.onnx'}:
                    archive.write(path, name)
        (records / 'environment.txt').write_text(capture([args.python, '-m', 'pip', 'freeze'], repo))
        if args.gpu != 'cpu':
            (records / 'gpu_before.txt').write_text(capture(['nvidia-smi'], repo))
            selected = capture(['nvidia-smi', '-i', args.gpu,
                                '--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu',
                                '--format=csv,noheader,nounits'], repo)
            (records / 'selected_gpu.txt').write_text(selected)
            fields = [field.strip() for field in selected.strip().split(',')]
            if len(fields) != 6 or int(fields[-2]) > 512 or int(fields[-1]) > 10:
                raise RuntimeError('Selected GPU is busy; no training started. Check selected_gpu.txt.')
            processes = capture(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                 '--format=csv,noheader'], repo)
            if any(line.split(',')[0].strip() == fields[1] for line in processes.splitlines()):
                raise RuntimeError('Selected GPU has a compute process; no training started.')
        runtime = subprocess.run([args.python, '-c',
            'import json,os,sys,torch; print(json.dumps(dict(python=sys.version,executable=sys.executable,'
            'torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),'
            'visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),cuda_available=torch.cuda.is_available(),'
            'matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,'
            'cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)));'
            ' print(torch.__config__.show())'], cwd=repo, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (records / 'runtime.txt').write_text(runtime.stdout)
        if runtime.returncode or (args.gpu != 'cpu' and '"cuda_available": true' not in runtime.stdout):
            raise RuntimeError('Runtime/CUDA probe failed; see runtime.txt')
        inputs = {}
        for index, flag in enumerate(native):
            key, sep, value = flag.partition('=')
            if key in {'--train_txt', '--val_txt', '--split_txt', '--hiera_checkpoint', '--pretrained', '--init_checkpoint', '--model'}:
                value = value if sep else native[index + 1]
                path = Path(value)
                path = path if path.is_absolute() else repo / path
                inputs[key] = {'path': str(path.resolve()), 'sha256': digest(path)}
        write_json(records / 'input_hashes.json', inputs)
        record['status'] = 'running'
        write_json(manifest, record)
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, start_new_session=True)
        def stop(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signum)
        for sig in [signal.SIGINT, signal.SIGTERM]:
            old_handlers[sig] = signal.signal(sig, stop)
        with (records / 'console.log').open('x') as console:
            for line in child.stdout:
                console.write(line)
                console.flush()
                print(line, end='', flush=True)
        code = child.wait()
        record.update(returncode=code, status='completed' if code == 0 else 'failed')
        text = (records / 'console.log').read_text()
        record['best_validation_iou'] = re.findall(r'Best val IoU: ([\d.]+) at epoch (\d+)', text)
        record['evaluation_metrics'] = dict(re.findall(r'^mean ([\w-]+): ([\d.]+)', text, re.M))
        record['artifacts'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                               for p in sorted(Path(paths['checkpoints']).glob('*.pth'))]
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        record.update(status='failed', error=str(error))
        raise
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
        record['ended_at'] = dt.datetime.now().astimezone().isoformat()
        write_json(manifest, record)
        # Per-run manifests are authoritative; locked append supports parallel GPUs.
        import fcntl
        with (records.parent / 'index.jsonl').open('a') as index:
            fcntl.flock(index, fcntl.LOCK_EX)
            index.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'Experiment record: {manifest}', flush=True)
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
