"""Strict CUDA execution and epoch-boundary fingerprints, without model changes."""
import hashlib
import json
import os
from pathlib import Path

import torch

from paired_inputs import append_record, state_digest

PROTOCOL = 'strict_cuda_v1'


def configure_runtime():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Launch with CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runtime_record():
    def sha(value):
        return hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
    return dict(protocol=PROTOCOL, deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        sdpa_enabled=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                          memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                          math=torch.backends.cuda.math_sdp_enabled()),
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        cpu_rng_sha256=sha(torch.get_rng_state()),
        cuda_rng_sha256=[sha(value) for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [])


def record_setup(model, seed, log_dir):
    record = dict(seed=seed, **runtime_record(), active_stochastic_modules=[
        name for name, module in model.named_modules()
        if module.training and 'Drop' in type(module).__name__
        and float(getattr(module, 'p', getattr(module, 'drop_prob', 0))) > 0])
    with (Path(log_dir) / 'numerical_runtime.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')


def record_phase_state(model, optimizer, scheduler, epoch, phase, log_dir):
    parameters = {name: parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad}
    moments = {name + '.' + key: value
               for name, parameter in model.named_parameters()
               for key, value in optimizer.state.get(parameter, {}).items()
               if torch.is_tensor(value)}
    append_record(Path(log_dir) / 'numerical_states.jsonl', dict(epoch=epoch, phase=phase,
        parameters_sha256=state_digest(parameters), buffers_sha256=state_digest(dict(model.named_buffers())),
        optimizer_sha256=state_digest(moments), scheduler=scheduler.state_dict(),
        learning_rates=[group['lr'] for group in optimizer.param_groups], **runtime_record()))


def record_selected_states(best_iou_state, best_loss_state, log_dir):
    with (Path(log_dir) / 'checkpoint_tensors.json').open('x') as stream:
        json.dump(dict(best_iou_sha256=state_digest(best_iou_state),
                       best_loss_sha256=state_digest(best_loss_state)), stream, indent=2)
        stream.write('\n')
