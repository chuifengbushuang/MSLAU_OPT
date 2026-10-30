"""Build one experiment and invoke its ordinary standalone launcher."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

from build_experiment import ROOT, build
from experiment_overrides import apply_overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    parser.add_argument('--seed', type=int, choices=[42, 1234, 2026], default=1234)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--split-dir', type=Path, help='directory containing train.txt and val.txt')
    parser.add_argument('--epochs', type=int, help='override maximum epochs; original default otherwise')
    parser.add_argument('--patience', type=int, default=0, help='validation IoU patience;0 disables early stopping')
    parser.add_argument('--fp32', action='store_true', help='FP32 with TF32 disabled; original code uses no AMP')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--hiera-checkpoint', type=Path)
    parser.add_argument('--run', action='store_true', help='without this flag: source validation and command preview only')
    args = parser.parse_args()
    if (args.epochs is not None and args.epochs < 1) or args.patience < 0:
        parser.error('epochs must be positive; patience must be nonnegative')
    if args.run:
        definition=ROOT/'experiments'/args.experiment
        config=json.loads((definition/'config.json').read_text())
        relevant=[ROOT/'experiments/exp00_baseline',definition,
                  ROOT/'experiments'/config.get('source_definition',args.experiment),
                  Path(__file__),Path(__file__).with_name('experiment_overrides.py')]
        if args.split_dir and args.split_dir.resolve().is_relative_to(ROOT):relevant.append(args.split_dir.resolve())
        if subprocess.check_output(['git','status','--porcelain','--',*map(str,relevant)],cwd=ROOT,text=True).strip():
            raise RuntimeError('Commit selected experiment/source changes before training; no training started')
    work = ROOT / '.work'
    work.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=args.experiment + '_', dir=work))
    source = stage / 'source'
    try:
        build(args.experiment, source)
        apply_overrides(source,split_dir=args.split_dir,epochs=args.epochs,patience=args.patience,
                        fp32=args.fp32,dataset=args.dataset,hiera_checkpoint=args.hiera_checkpoint)
        command = [sys.executable, '-u', str(source / 'run.py'), '--seed', str(args.seed), '--gpu', str(args.gpu)]
        if args.run:
            command.append('--run')
        with subprocess.Popen(command, cwd=source) as process:
            def stop(signum, frame):
                if process.poll() is None:
                    process.send_signal(signum)
            previous = {sig: signal.signal(sig, stop) for sig in [signal.SIGINT, signal.SIGTERM]}
            try:
                return process.wait()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    finally:
        shutil.rmtree(stage)


if __name__ == '__main__':
    raise SystemExit(main())
