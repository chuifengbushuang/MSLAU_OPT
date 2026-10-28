"""FADC FreqSelect adapted to the existing NAF shell, without dynamic sampling.

Reference: Linwei-Chen/FADC 923102e6344d9b90ec427cb15201efd514bf316e.
Four square frequency bands; the lowest band passes unchanged. Three 3x3
spatial gates use 2*sigmoid and zero initialization. Both streams share gates.
Our frequency masks use normalized fftfreq rather than rounded image slices,
so odd/even rectangular grids preserve conjugacy. H/G/FS are defined below.
"""
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=32)
def lowpass_masks(height, width, device, dtype):
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype).abs()[:, None]
    fx = torch.fft.rfftfreq(width, device=device, dtype=dtype).abs()[None, :]
    radius = torch.maximum(fy, fx)
    return tuple(radius < cutoff for cutoff in (.25, .125, .0625))


class BandSelection(nn.Module):
    def __init__(self, channels, spectral):
        super().__init__()
        self.spectral = spectral
        with torch.random.fork_rng(devices=[]):
            self.gates = nn.ModuleList([nn.Conv2d(channels, 1, 3, padding=1) for _ in range(3)])
            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)

    def decompose(self, x):
        if not self.spectral:
            # Matched generic spatial-attention control; no FFT or smoothing.
            # Four identical normalized branches sum to the original feature.
            return (x * .25,) * 4
        spectrum = torch.fft.rfft2(x, norm='ortho')
        masks = lowpass_masks(*x.shape[-2:], x.device, x.dtype)
        lows = [torch.fft.irfft2(spectrum * mask, s=x.shape[-2:], norm='ortho') for mask in masks]
        return x - lows[0], lows[0] - lows[1], lows[1] - lows[2], lows[2]

    def forward(self, x):
        bands = self.decompose(x)
        return sum(2 * gate(x).sigmoid() * band for gate, band in zip(self.gates, bands[:3])) + bands[3]


def summary(x):
    mean = x.mean((-2, -1), keepdim=True)
    rms = ((x - mean).square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
    return torch.cat((mean, rms), 1)


class SelectionStage(nn.Module):
    def __init__(self, legacy, mode):
        super().__init__()
        if mode not in ('h', 'g', 'fs'):
            raise ValueError(mode)
        self.local_projection = legacy.local_projection
        self.context_projection = legacy.context_projection
        self.fusion_gate = legacy.fusion_gate
        self.output_projection = legacy.output_projection
        self.filter = nn.Identity() if mode == 'h' else BandSelection(legacy.local_projection.out_channels, mode == 'fs')

    def forward(self, local, context):
        dtype = local.dtype
        with torch.autocast(device_type=local.device.type, enabled=False):
            work = local if dtype == torch.float64 else local.float()
            coarse = F.interpolate(context.to(work.dtype), size=local.shape[-2:], mode='bilinear', align_corners=True)
            left = self.filter(self.local_projection(work))
            right = self.filter(self.context_projection(coarse))
            descriptor = torch.cat((summary(left), summary(right)), 1)
            gate = self.fusion_gate(descriptor).reshape(work.shape[0], 2, work.shape[1], 1, 1).softmax(1)
            residual = self.output_projection(gate[:, 0] * left + gate[:, 1] * right)
            output = work + residual
            with torch.no_grad():
                weight = gate[:, 1]
                ratio = residual.square().mean().sqrt() / work.square().mean().sqrt().clamp_min(1e-6)
                diagnostics = torch.stack((weight.mean(), weight.std(1, unbiased=False).mean(),
                                           weight.std(0, unbiased=False).mean(), ratio))
        return output.to(dtype), diagnostics


class CascadeSelection(nn.Module):
    def __init__(self, legacy, mode):
        super().__init__()
        self.stages = nn.ModuleDict({name: SelectionStage(stage, mode) for name, stage in legacy.stages.items()})

    def forward(self, features):
        enhanced, context, diagnostics = list(features), features[3], {}
        for level in (3, 2, 1):
            context, diagnostics[level] = self.stages[f'p{level}'](features[level - 1], context)
            enhanced[level - 1] = context
        return (enhanced, features[0].new_zeros((), dtype=torch.float32),
                features[0].new_empty((0,), dtype=torch.float32),
                torch.stack([diagnostics[i] for i in (1, 2, 3)]))
