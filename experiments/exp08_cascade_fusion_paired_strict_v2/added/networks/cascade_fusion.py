"""Experiment 8: experiment-7 cascade/projections/gates, without frequency processing.

Standalone runtime source: no imports from experiment 7, FFT, selector, alpha
or regularizer. Shared layers retain exactly the experiment-7 initialization.
"""
import torch
from torch import nn
from torch.nn import functional as F


FUSION_STATS_NAMES = ('context_weight', 'channel_gate_std', 'image_gate_std',
                      'residual_ratio')


def spatial_summary(features):
    mean = features.mean((-2, -1), keepdim=True)
    rms = ((features - mean).square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
    return torch.cat([mean, rms], dim=1)


class DualInputFusion2D(nn.Module):
    def __init__(self, level, channels=64):
        super().__init__()
        if level not in (1, 2, 3):
            raise ValueError('Only fusion levels 1, 2 and 3 exist in experiment 8')
        self.level = level
        with torch.random.fork_rng(devices=[]):
            self.local_projection = nn.Conv2d(channels, channels, 1, bias=False)
            self.context_projection = nn.Conv2d(channels, channels, 1, bias=False)
            # Exp7 initialized this selector BEFORE the fusion gate. Consume
            # exactly its CPU draws here, then discard it: deleting these draws
            # would silently change shared gate/projection initialization.
            # This is initialization only, never registered or run in forward.
            _discarded_selector = nn.Sequential(
                nn.Conv2d(5, 16, 1), nn.GELU(), nn.Conv2d(16, 2, 1))
            self.fusion_gate = nn.Sequential(nn.Conv2d(4 * channels, 16, 1), nn.GELU(),
                                             nn.Conv2d(16, 2 * channels, 1))
            self.output_projection = nn.Conv2d(channels, channels, 1, bias=False)
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.zeros_(self.fusion_gate[-1].bias)
            nn.init.zeros_(self.output_projection.weight)

    def forward(self, local, context):
        original_dtype = local.dtype
        # Match exp7's FP32 projection/fusion arithmetic under autocast.
        with torch.autocast(device_type=local.device.type, enabled=False):
            work = local if original_dtype == torch.float64 else local.float()
            coarse = context.to(work.dtype)
            coarse = F.interpolate(coarse, size=local.shape[-2:], mode='bilinear', align_corners=True)
            projected_local = self.local_projection(work)
            projected_context = self.context_projection(coarse)
            # Identity frequency path: directly summarize and fuse projections.
            descriptor = torch.cat([spatial_summary(projected_local), spatial_summary(projected_context)], dim=1)
            logits = self.fusion_gate(descriptor).reshape(local.shape[0], 2, local.shape[1], 1, 1)
            gate = logits.softmax(dim=1)
            fused = gate[:, 0] * projected_local + gate[:, 1] * projected_context
            residual = self.output_projection(fused)
            output = work + residual
            with torch.no_grad():
                context_weight = gate[:, 1]
                ratio = residual.square().mean().sqrt() / work.square().mean().sqrt().clamp_min(1e-6)
                fusion_stats = torch.stack([context_weight.mean(),
                    context_weight.std(dim=1, unbiased=False).mean(),
                    context_weight.std(dim=0, unbiased=False).mean(), ratio])
        return output.to(original_dtype), fusion_stats


class CascadeFusion2D(nn.Module):
    """F3=Fusion3(P3,P4); F2=Fusion2(P2,F3); F1=Fusion1(P1,F2)."""

    def __init__(self, channels=64):
        super().__init__()
        self.stages = nn.ModuleDict({
            'p%d' % level: DualInputFusion2D(level, channels) for level in (1, 2, 3)
        })

    def forward(self, features):
        if len(features) != 4:
            raise ValueError('Expected the four original RFB outputs P1,P2,P3,P4')
        enhanced = list(features)
        context = features[3]
        fusion_stats = {}
        for level in (3, 2, 1):
            context, stats = self.stages['p%d' % level](features[level - 1], context)
            enhanced[level - 1] = context
            fusion_stats[level] = stats
        return enhanced, torch.stack([fusion_stats[i] for i in (1, 2, 3)])
