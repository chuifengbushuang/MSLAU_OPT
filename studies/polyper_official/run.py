"""Execute author CLI preflight, one full 80k run, and official test evaluation."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

from prepare import ROOT, WORK, OUT, SOURCE, FRAMEWORK, SIZES, sha, write


def environment():
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(FRAMEWORK), str(Path(__file__).parent)])
    env['CUDA_VISIBLE_DEVICES'] = '0'
    # Official MMEngine 0.10.7 predates PyTorch's weights_only=True default.
    # Only the recorded upstream/local checkpoints are loaded in this study.
    env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    env['MPLCONFIGDIR'] = str(WORK / 'matplotlib')
    env['PYTHONUNBUFFERED'] = '1'
    return env


def verify_sources():
    manifest = json.loads((OUT / 'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(SOURCE / name) == digest, f'Author source changed: {name}'
    for name, digest in manifest['added_sources'].items():
        assert sha(Path(__file__).parent / name) == digest, f'Study implementation changed: {name}'
    assert sha(WORK / 'official_80k.py') == manifest['overlay_sha256']
    return sha(OUT / 'source_manifest.json')


def call(script, config, directory, log, extra=()):
    command = [sys.executable, str(FRAMEWORK / 'tools' / script), str(config),
               *map(str, extra), '--work-dir', str(directory)]
    write(OUT / 'process.json', dict(status='running', command=command,
          parent_pid=os.getpid(), log=str(log), phase=log.stem))
    with log.open('a') as stream:
        stream.write('\nCOMMAND ' + json.dumps(command) + '\n')
        stream.flush()
        child = subprocess.Popen(command, cwd=FRAMEWORK, env=environment(),
                                 stdout=stream, stderr=subprocess.STDOUT)
        write(OUT / 'process.json', dict(status='running', command=command,
              parent_pid=os.getpid(), child_pid=child.pid, log=str(log), phase=log.stem))
        code = child.wait()
    if code:
        raise RuntimeError(f'Official {script} exit {code}: {log}')


def preflight():
    signature = verify_sources()
    directory = WORK / 'smoke'
    assert not directory.exists(), 'Inspect existing preflight instead of silently replacing it'
    call('train.py', WORK / 'smoke.py', directory, OUT / 'smoke_train.log')
    state = torch.load(directory / 'iter_2.pth', map_location='cpu', weights_only=False)
    assert state['meta']['iter'] == 2 and 'optimizer' in state and 'param_schedulers' in state
    del state
    call('train.py', WORK / 'smoke_resume.py', directory,
         OUT / 'smoke_resume.log', ['--resume'])
    state = torch.load(directory / 'iter_3.pth', map_location='cpu', weights_only=False)
    assert state['meta']['iter'] == 3
    steps = [int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
    assert steps and set(steps) == {3}
    del state
    call('test.py', WORK / 'smoke.py', WORK / 'smoke_test', OUT / 'smoke_test.log',
         [directory / 'iter_3.pth', '--out', WORK / 'smoke_predictions'])
    assert len(list((WORK / 'smoke_predictions').glob('*.png'))) == 2
    test = json.loads((WORK / 'smoke_test/official_test_metrics.json').read_text())
    assert all(np.isfinite(test['metrics'][key]) for key in ('mIoU', 'mDice'))
    write(OUT / 'preflight.json', dict(status='passed', source_signature=signature,
          official_train_updates=3, batch_size=6, optimizer_resume_step=3,
          official_test_images=2, evaluation=test, author_source_unmodified=True))
    print('Official train/resume/test preflight passed', flush=True)


def diagnostics(name, predictions, official):
    rows, totals = [], np.zeros(4, dtype=np.int64)
    gt_dir = WORK / 'data/TestDataset' / name / 'masks'
    assert len(list(predictions.glob('*.png'))) == SIZES[name]
    for path in sorted(gt_dir.glob('*.png')):
        gt, pred = np.asarray(Image.open(path)), np.asarray(Image.open(predictions / path.name))
        assert gt.shape == pred.shape and set(np.unique(pred)) <= {0, 1}
        gt, pred = gt.astype(bool), pred.astype(bool)
        tp, fp, fn, tn = (int(a.sum()) for a in
                         (gt & pred, ~gt & pred, gt & ~pred, ~gt & ~pred))
        totals += [tp, fp, fn, tn]
        rows.append(dict(id=path.stem, tp=tp, fp=fp, fn=fn, tn=tn,
                         iou=tp / (tp + fp + fn), dice=2 * tp / (2 * tp + fp + fn)))
    tp, fp, fn, tn = map(int, totals)
    fg_iou, bg_iou = tp / (tp + fp + fn), tn / (tn + fp + fn)
    global_iou = (fg_iou + bg_iou) * 50
    global_dice = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) * 50
    assert abs(global_iou - official['metrics']['mIoU']) <= .00501
    assert abs(global_dice - official['metrics']['mDice']) <= .00501
    result = dict(dataset=name, images=len(rows), official=official['metrics'],
                  foreground_image_mean_iou_percent=float(np.mean([r['iou'] for r in rows]) * 100),
                  foreground_image_mean_dice_percent=float(np.mean([r['dice'] for r in rows]) * 100),
                  foreground_global_iou_percent=fg_iou * 100,
                  background_global_iou_percent=bg_iou * 100,
                  exact_official_class_mean_iou_percent=global_iou,
                  exact_official_class_mean_dice_percent=global_dice,
                  threshold=.3, native_image_geometry=True, totals=dict(tp=tp, fp=fp, fn=fn, tn=tn),
                  per_image=rows)
    write(OUT / f'evaluation_{name}.json', result)
    return {k: v for k, v in result.items() if k != 'per_image'}


def evaluate():
    verify_sources()
    checkpoint = WORK / 'formal/iter_80000.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['meta']['iter'] == 80000
    steps = [int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
    assert steps and set(steps) == {80000}
    del state
    summary = []
    for name in SIZES:
        directory = WORK / 'evaluation' / name
        predictions = WORK / 'predictions' / name
        config = WORK / f'test_{name}.py'
        config.write_text(f'_base_ = {str(WORK / "official_80k.py")!r}\n'
            f'test_dataloader = dict(dataset=dict(data_prefix=dict(img_path={str(WORK / "data/TestDataset" / name / "images")!r}, '
            f'seg_map_path={str(WORK / "data/TestDataset" / name / "masks")!r})))\n')
        if not (directory / 'official_test_metrics.json').exists():
            call('test.py', config, directory, OUT / f'test_{name}.log',
                 [checkpoint, '--out', predictions])
        official = json.loads((directory / 'official_test_metrics.json').read_text())
        assert official['checkpoint'] == str(checkpoint)
        summary.append(diagnostics(name, predictions, official))
    result = dict(status='completed', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                  completed_updates=80000, optimizer_updates=80000, seed=1234,
                  selection='Final iter_80000; no best-test checkpoint selection',
                  author_kvasir_iou_percent=90.36, author_kvasir_dice_percent=94.82,
                  evaluations=summary, source_signature=verify_sources())
    result['kvasir_iou_delta_percentage_points'] = summary[0]['official']['mIoU'] - 90.36
    write(OUT / 'result.json', result)
    lines = ['# Polyper 官方公开配置 80k 对照', '',
             '完整训练 80,000 步，固定最终权重；下表使用作者原始 IoUMetric。', '',
             '| 数据集 | 数量 | 官方 mIoU % | 官方 mDice % | 前景逐图 IoU % |',
             '|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["dataset"]} | {r["images"]} | {r["official"]["mIoU"]:.2f} | '
                     f'{r["official"]["mDice"]:.2f} | {r["foreground_image_mean_iou_percent"]:.4f} |')
    lines += ['', f'论文 Kvasir mIoU 90.36%；本次差 {result["kvasir_iou_delta_percentage_points"]:+.2f} 个百分点。',
              '', '公开主配置 lr=6e-5、wd=0.01 与论文文字不同；硬件/依赖版本不同，详见运行清单及源码 README。',
              '前景逐图指标也使用本次官方阈值 0.3、原始尺寸，不等同于此前 352/0.5 的统一协议。']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return result


def resume_uniform_queue():
    path = ROOT / 'analysis/20261030_paper_recipes/official_polyper_priority_hold.json'
    hold = json.loads(path.read_text())
    resumed = []
    # Only resume recorded, still-existing processes with matching study argv.
    for pid in hold.get('stopped_pids', []) + [hold['supervisor_pid']]:
        proc = Path('/proc') / str(pid)
        if proc.exists():
            argv = (proc / 'cmdline').read_bytes().decode().replace('\x00', ' ')
            assert 'paper_recipe_study.py' in argv or 'run_paper_recipe_study.py' in argv
            os.kill(pid, 18)  # SIGCONT on Linux
            resumed.append(pid)
    hold.update(resumed_after_official_success=True, resumed_pids=resumed)
    write(path, hold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'run', 'evaluate'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'official.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if args.mode == 'preflight':
            preflight()
        elif args.mode == 'run':
            gate = json.loads((OUT / 'preflight.json').read_text())
            assert gate['status'] == 'passed' and gate['source_signature'] == verify_sources()
            assert not (WORK / 'formal').exists(), 'Inspect existing formal run; never silently restart'
            import mmcv, mmengine
            write(OUT / 'run_manifest.json', dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  torch=torch.__version__, torch_cuda=torch.version.cuda, mmcv=mmcv.__version__,
                  mmengine=mmengine.__version__, python=sys.version, gpu=torch.cuda.get_device_name(0),
                  seed=1234, source_signature=verify_sources(),
                  packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                  weight_loading_compatibility='TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for recorded official/local checkpoints'))
            call('train.py', WORK / 'official_80k.py', WORK / 'formal', OUT / 'train_80k.log')
            evaluate()
            resume_uniform_queue()
        else:
            evaluate()
        write(OUT / 'process.json', dict(status='completed', phase=args.mode))
    except Exception as exc:
        write(OUT / 'failure.json', dict(phase=args.mode, error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
