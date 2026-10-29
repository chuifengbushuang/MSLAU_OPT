"""Mask-only FAME-Net adaptation on experiment-9's raw local residual.

No frequency/selection method or state key of the original stage is changed.
The bound forward is installed once at model construction; filter_pair and all
selection/context methods remain the original implementation.
"""
import types

import torch
from torch import nn
from torch.nn import functional as F


class SemanticResidualGate(nn.Module):
    """Paper-inspired multiscale mask, unit initialization via 2*sigmoid.

    Hidden width 16 and dilation 2 are explicit adaptation choices, not reported
    paper defaults. No normalization, extra activation, attention or new loss.
    """
    def __init__(self, channels=64, hidden=16, dilation=2):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.standard = nn.Conv2d(channels, hidden, 3, padding=1)
            self.dilated = nn.Conv2d(channels, hidden, 3, padding=dilation, dilation=dilation)
            self.merge = nn.Conv2d(hidden, hidden, 1)
            self.shortcut = nn.Conv2d(channels, hidden, 1)
            self.mask = nn.Conv2d(hidden, 1, 1)
            nn.init.zeros_(self.mask.weight)
            nn.init.zeros_(self.mask.bias)

    def forward(self, features):
        combined = self.merge(self.standard(features) + self.dilated(features)) + self.shortcut(features)
        return 2 * self.mask(combined).sigmoid()


def _gated_forward(self, local, context):
    original_dtype = local.dtype
    with torch.autocast(device_type=local.device.type, enabled=False):
        work = local if original_dtype == torch.float64 else local.float()
        coarse = F.interpolate(context.to(work.dtype), size=local.shape[-2:],
                               mode='bilinear', align_corners=True)
        local_filtered, context_filtered, penalties, stats = self.filter_pair(
            self.local_projection(work), self.context_projection(coarse))
        # Identical GAP + centered RMS and channel fusion as the legacy stage.
        def summary(x):
            mean = x.mean((-2, -1), keepdim=True)
            rms = ((x - mean).square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
            return torch.cat([mean, rms], dim=1)
        descriptor = torch.cat([summary(local_filtered), summary(context_filtered)], dim=1)
        gate = self.fusion_gate(descriptor).reshape(work.shape[0], 2, work.shape[1], 1, 1).softmax(1)
        residual = self.output_projection(gate[:, 0] * local_filtered + gate[:, 1] * context_filtered)
        mask = self.residual_gate(coarse if self.residual_gate_source == 'deep' else work)
        output = mask * work + residual
        with torch.no_grad():
            context_weight = gate[:, 1]
            ratio = residual.square().mean().sqrt() / work.square().mean().sqrt().clamp_min(1e-6)
            fusion = torch.stack([context_weight.mean(), context_weight.std(1, unbiased=False).mean(),
                                  context_weight.std(0, unbiased=False).mean(), ratio])
    return output.to(original_dtype), penalties, stats, fusion


def install_residual_gates(cascade, mode):
    if mode not in ('original', 'local', 'deep'):
        raise ValueError(mode)
    if mode == 'original':
        return cascade
    for stage in cascade.stages.values():
        if hasattr(stage, 'residual_gate'):
            raise RuntimeError('Residual gate already installed')
        stage.residual_gate = SemanticResidualGate(stage.local_projection.in_channels)
        stage.residual_gate_source = mode
        stage.forward = types.MethodType(_gated_forward, stage)
    return cascade
