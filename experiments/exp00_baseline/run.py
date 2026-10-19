"""Fixed original P5.1 experiment. Preview by default; --run starts training."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
OUTPUT = Path(os.environ.get('P51_OUTPUT_ROOT', '/data/models/le/mslau-net'))
DATASET = '/data/dataset/zsy/mslau-net/Kvasir-SEG'
HIERA = '/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_source():
    path = HERE / 'build_manifest.json'
    manifest = json.loads((path if path.exists() else HERE / 'source_manifest.json').read_text())
    for name, expected in {**manifest['files'], **manifest.get('controls', {})}.items():
        if sha256(HERE / name) != expected:
            raise RuntimeError(f'Original experiment0 file changed: {name}')
    return manifest


def commands(seed, run_id):
    config = json.loads((HERE / 'config.json').read_text())
    checkpoints = OUTPUT / 'checkpoints' / run_id
    records = OUTPUT / 'experiments' / run_id
    train = [PYTHON, '-u', str(HERE / 'kvasir_train.py'),
        '--dataset', DATASET, '--train_txt', str(HERE / 'configs/splits/kvasir/train.txt'),
        '--val_txt', str(HERE / 'configs/splits/kvasir/val.txt'),
        '--output_dir', str(checkpoints), '--log_dir', str(records / 'logs'), '--seed', str(seed)]
    for flag, value in config['train_options'].items():
        if flag in {'dataset', 'train_txt', 'val_txt', 'output_dir', 'log_dir', 'seed'}:
            raise ValueError(f'Managed train path/seed cannot be overridden in config: {flag}')
        train.append('--' + flag)
        if value is not None:
            train.append(str(value))
    evaluation = [PYTHON, '-u', str(HERE / 'kvasir_test.py'), '--dataset', DATASET,
        '--split_txt', str(HERE / 'configs/splits/kvasir/val.txt'),
        '--model', str(checkpoints / 'kvasir_best_model.pth'),
        '--debug_dir', str(OUTPUT / 'visualization' / run_id)]
    for flag, value in config['eval_options'].items():
        if flag in {'dataset', 'split_txt', 'model', 'debug_dir'}:
            raise ValueError(f'Managed evaluation path cannot be overridden in config: {flag}')
        evaluation.append('--' + flag)
        if value is not None:
            evaluation.append(str(value))
    return train, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[42, 1234, 2026], default=1234)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('--gpu must be a nonnegative physical GPU index')
    source = verify_source()
    config = json.loads((HERE / 'config.json').read_text())
    run_id = f"{config['id']}_seed{args.seed}_{datetime.datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d_%H%M%S}_{os.getpid()}"
    train, evaluation = commands(args.seed, run_id)
    print('Source verified:', config['id'], 'baseline reference', source['source_commit'])
    print('Train:', shlex.join(train), flush=True)
    print('Evaluate:', shlex.join(evaluation), flush=True)
    if not args.run:
        print('Preview only. No output directories created and no training started.')
        return
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    if not Path(PYTHON).is_file() or not Path(DATASET).is_dir() or not Path(HIERA).is_file():
        raise RuntimeError('Missing cloned environment, dataset or Hiera checkpoint')
    gpu = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    gpu_uuid, memory, utilization = [part.strip() for part in gpu.strip().split(',')]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'], text=True)
    if int(memory) > 512 or int(utilization) > 10 or any(
            line.split(',')[0].strip() == gpu_uuid for line in processes.splitlines()):
        raise RuntimeError(f'Selected GPU is busy. No training started.\n{gpu}')
    records = OUTPUT / 'experiments' / run_id
    records.mkdir(parents=True, exist_ok=False)
    (OUTPUT / 'checkpoints' / run_id).mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), NO_ALBUMENTATIONS_UPDATE='1')
    metadata = dict(run_id=run_id, seed=args.seed, gpu=args.gpu, source=source, status='running',
                    config=config, train=train, evaluation=evaluation, checkpoint=str(OUTPUT / 'checkpoints' / run_id / 'kvasir_best_model.pth'))
    metadata['reproduce_from_repository'] = [PYTHON, 'tools/run_experiment.py', config['id'],
                                              '--seed', str(args.seed), '--gpu', str(args.gpu), '--run']
    (records / 'command.txt').write_text(shlex.join(train) + '\n')
    (records / 'evaluation_command.txt').write_text(shlex.join(evaluation) + '\n')
    (records / 'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    try:
        runtime = subprocess.check_output([PYTHON, '-c',
            'import sys,torch,torchvision; assert torch.cuda.is_available(); '
            'print(sys.version,torch.__version__,torchvision.__version__,torch.version.cuda,torch.backends.cudnn.version()); '
            'print(torch.cuda.get_device_name(0)); x=torch.randn(8,device="cuda",requires_grad=True); '
            'x.square().sum().backward(); assert torch.isfinite(x.grad).all()'], env=environment, text=True)
        (records / 'runtime.txt').write_text(runtime)
        (records / 'environment.txt').write_text(subprocess.check_output(
            [PYTHON, '-m', 'pip', 'freeze'], text=True, env=environment))
        metadata['hiera_sha256'] = sha256(HIERA)
        (records / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
        for command, log in [(train, 'console.log'), (evaluation, 'evaluation.log')]:
            with (records / log).open('x') as stream:
                with subprocess.Popen(command, cwd=HERE, env=environment, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
                    try:
                        for line in process.stdout:
                            stream.write(line)
                            stream.flush()
                            print(line, end='', flush=True)
                        if process.wait():
                            raise RuntimeError(f'Failed: {log}; see {records}')
                    except BaseException:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                        raise
        metadata['checkpoint_sha256'] = sha256(metadata['checkpoint'])
        best = re.search(r'Best val IoU: ([\d.]+) at epoch (\d+)', (records / 'console.log').read_text())
        if best:
            metadata.update(best_val_iou=float(best[1]), best_epoch=int(best[2]))
        metadata['evaluation_metrics'] = {key: float(value) for key, value in re.findall(
            r'^mean ([\w-]+): ([\d.]+)', (records / 'evaluation.log').read_text(), re.M)}
        # Preserve the original script's three checkpoints and filenames.
        metadata['checkpoint_files'] = {path.name: sha256(path)
            for path in sorted((OUTPUT / 'checkpoints' / run_id).glob('*.pth'))}
        metadata['status'] = 'completed'
    except BaseException as error:
        metadata.update(status='failed', error=str(error))
        raise
    finally:
        (records / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print('Records:', records, flush=True)


if __name__ == '__main__':
    main()
