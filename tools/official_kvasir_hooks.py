"""Budget and recording hooks added only to new official-split source snapshots."""
import json
import logging
import math
import os
from pathlib import Path
from itertools import islice


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best = float('-inf')
        self.bad_epochs = 0

    def update(self, value):
        if not math.isfinite(value):
            raise ValueError('Nonfinite validation IoU')
        if value > self.best:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def configure_fp32():
    import torch
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.is_autocast_enabled():
        raise RuntimeError('This study requires FP32 without autocast')


class LimitedLoader:
    """Only preflight runs limit batches; formal runs receive original loaders."""
    def __init__(self, loader, count):
        self.loader, self.count = loader, count

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        return islice(iter(self.loader), self.count)

    def __len__(self):
        return min(len(self.loader), self.count)


def budget_loaders(loaders):
    steps = int(os.environ.get('OFFICIAL_KVASIR_SMOKE_STEPS', '0'))
    return {k: LimitedLoader(v, steps) for k, v in loaders.items()} if steps else loaders


_stopper = None


def after_epoch(epoch, accuracy_list, loss_list, model, optimizer, scheduler, save_dir):
    global _stopper
    import torch
    records = Path(os.environ['OFFICIAL_KVASIR_RECORDS'])
    if _stopper is None:
        _stopper = EarlyStopping(int(os.environ.get('OFFICIAL_KVASIR_PATIENCE', '10')))
    score = accuracy_list['valid'][-1]
    stopped = _stopper.update(score)
    dtypes = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
    if dtypes != ['torch.float32']:
        raise RuntimeError(f'Unexpected parameter precision: {dtypes}')
    runtime_path = records / 'runtime.json'
    if not runtime_path.exists():
        write(runtime_path, dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), python=__import__('sys').version,
            device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu fixture',
            precision='FP32', parameter_dtypes=dtypes,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            autocast=torch.is_autocast_enabled()))
    row = dict(completed_epochs=epoch + 1, train_iou=accuracy_list['train'][-1],
        val_iou=score, train_main_loss=loss_list['train'][-1], val_main_loss=loss_list['valid'][-1],
        best_val_iou=_stopper.best, bad_epochs=_stopper.bad_epochs, patience=_stopper.patience,
        learning_rates=[g['lr'] for g in optimizer.param_groups], scheduler=scheduler.state_dict(),
        parameter_dtypes=dtypes, autocast=torch.is_autocast_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        max_epochs=int(os.environ['OFFICIAL_KVASIR_MAX_EPOCHS']),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    with (records / 'budget_epochs.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    write(records / 'progress.json', dict(status='early_stopped' if stopped else 'training', **row))
    logging.info('Official split budget: %s', row)
    return stopped


def record_evaluation(metrics):
    import statistics
    if not metrics or any(len(v) != 120 for v in metrics.values()):
        raise RuntimeError('Official evaluation must contain exactly120 images per metric')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for v in metrics.values() for x in v):
        raise RuntimeError('Invalid evaluation metric')
    values = {k: statistics.mean(v) for k, v in metrics.items()}
    write(Path(os.environ['OFFICIAL_KVASIR_RECORDS']) / 'evaluation.json',
          dict(metrics=values, images=len(metrics['iou']), per_image_metrics=metrics,
               scope='Official120 used for early stopping and reporting; no independent third test set'))
