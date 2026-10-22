"""Build the new baseline and audit real inputs in isolated CPU processes."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from build_experiment import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='/data/dataset/zsy/mslau-net/Kvasir-SEG')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or (Path(os.environ.get('P51_OUTPUT_ROOT', '/data/models/le/mslau-net')) /
                            'experiments' / ('paired_aug_v1_input_check_' +
                            datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}.json'))
    with tempfile.TemporaryDirectory(prefix='p51_paired_input_check_') as temporary:
        source = Path(temporary) / 'source'
        build('exp00_baseline_paired_aug_v1', source)
        subprocess.run([sys.executable, str(source / 'check_inputs.py'), '--dataset', args.dataset,
                        '--seed', str(args.seed), '--workers', str(args.workers),
                        '--epochs', str(args.epochs), '--output', str(output.resolve())],
                       cwd=source, check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES='', NO_ALBUMENTATIONS_UPDATE='1'))


if __name__ == '__main__':
    main()
