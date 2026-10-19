"""Build one experiment and invoke its ordinary standalone launcher."""
import argparse
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

from build_experiment import ROOT, build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    parser.add_argument('--seed', type=int, choices=[42, 1234, 2026], default=1234)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--run', action='store_true', help='without this flag: source validation and command preview only')
    args = parser.parse_args()
    if args.run and subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit experiment changes before training; no training started')
    work = ROOT / '.work'
    work.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=args.experiment + '_', dir=work))
    source = stage / 'source'
    try:
        build(args.experiment, source)
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
