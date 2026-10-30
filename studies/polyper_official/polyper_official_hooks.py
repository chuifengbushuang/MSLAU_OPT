"""Observational records around the unmodified official MMEngine runner."""
import json
from pathlib import Path
import time

from mmengine.hooks import Hook
from mmseg.registry import HOOKS
import torch


@HOOKS.register_module()
class OfficialProtocolRecordHook(Hook):
    priority = 'LOWEST'

    def __init__(self, formal=True):
        self.formal = formal

    def record(self, runner, **values):
        path = Path(runner.work_dir) / 'protocol_progress.json'
        values.update(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      completed_iters=runner.iter, max_iters=runner.max_iters,
                      formal=self.formal)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(values, indent=2, ensure_ascii=False) + '\n')
        tmp.replace(path)

    def before_train(self, runner):
        cfg = runner.cfg
        if self.formal:
            assert runner.max_iters == 80000
            assert cfg.train_cfg.val_interval == 8000
            assert cfg.train_dataloader.batch_size == 6
            assert len(runner.train_dataloader.dataset.dataset) == 1450
            assert cfg.optim_wrapper.optimizer.lr == .00006
            assert cfg.optim_wrapper.optimizer.weight_decay == .01
            assert cfg.param_scheduler[1].end == 160000
            assert cfg.model.decode_head.image_size == 128
            assert cfg.crop_size == (224, 224)
        self.started = time.monotonic()
        self.initial_iter = runner.iter
        # Verify the actual initialized backbone against every compatible
        # tensor in the official ImageNet checkpoint, before any update.
        if not runner.cfg.resume:
            original = torch.load(cfg.model.backbone.init_cfg.checkpoint,
                                  map_location='cpu', weights_only=False)
            original = original.get('state_dict', original)
            state = runner.model.backbone.state_dict()
            matched = [k for k, v in original.items() if k in state and v.shape == state[k].shape]
            assert len(matched) >= 181
            assert all(torch.equal(original[k], state[k].cpu()) for k in matched)
            (Path(runner.work_dir) / 'pretraining_audit.json').write_text(json.dumps(
                dict(equal_tensors=len(matched), equal_elements=sum(original[k].numel() for k in matched)), indent=2))
        self.record(runner, status='training', initial_iter=self.initial_iter)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        done = runner.iter + 1
        if done <= 3 or done % 50 == 0 or done == runner.max_iters:
            loss = float(outputs['loss'].detach().cpu())
            assert torch.isfinite(outputs['loss']).all(), 'Nonfinite official training loss'
            self.record(runner, status='training', completed_update=done, loss=loss,
                        elapsed_seconds=time.monotonic() - self.started,
                        learning_rates=[g['lr'] for g in runner.optim_wrapper.optimizer.param_groups[:1]],
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)

    def after_val_epoch(self, runner, metrics=None):
        path = Path(runner.work_dir) / 'official_validation.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(dict(completed_iters=runner.iter, metrics=metrics)) + '\n')

    def after_train(self, runner):
        self.record(runner, status='training_finished', elapsed_seconds=time.monotonic() - self.started)

    def after_test_epoch(self, runner, metrics=None):
        (Path(runner.work_dir) / 'official_test_metrics.json').write_text(json.dumps(
            dict(metrics=metrics, checkpoint=runner.cfg.load_from,
                 evaluator='unmodified mmseg.evaluation.metrics.IoUMetric',
                 classes=['background', 'foreground']), indent=2) + '\n')
