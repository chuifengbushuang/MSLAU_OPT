"""FreDFT MFDA/FDFFL adapted inside the existing three-stage NAF shell.

Reference: WenCongWu/FreDFT d275f96eca0090e2267c6b24ae01361bc98850c0,
models/common.py: FDCA, FDFFN, FDFTM. Retains un-conjugated spectral product,
default FFT normalization, Q/K/V width 2C, shared weights between streams,
FFN expansion 3, and channel-wise LayerNorm. No LFEM/CGMM/final concat fuse.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = x.var(1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class MFDA(nn.Module):
    def __init__(self, channels, spectral):
        super().__init__()
        self.spectral = spectral
        self.norm1 = ChannelLayerNorm(channels)
        self.to_hidden = nn.Conv2d(channels, 6 * channels, 1, bias=False)
        self.to_hidden_dw = nn.Conv2d(6 * channels, 6 * channels, 3, padding=1,
                                    groups=6 * channels, bias=False)
        self.norm = ChannelLayerNorm(2 * channels)
        self.project_out = nn.Conv2d(2 * channels, channels, 1, bias=False)

    def response(self, q, k):
        if self.spectral:
            return torch.fft.irfft2(torch.fft.rfft2(q) * torch.fft.rfft2(k), s=q.shape[-2:])
        return q * k

    def forward(self, local, context):
        lq, lk, lv = self.to_hidden_dw(self.to_hidden(self.norm1(local))).chunk(3, 1)
        cq, ck, cv = self.to_hidden_dw(self.to_hidden(self.norm1(context))).chunk(3, 1)
        return (local + self.project_out(self.norm(self.response(lq, lk)) * cv),
                context + self.project_out(self.norm(self.response(cq, ck)) * lv))


class FDFFL(nn.Module):
    def __init__(self, channels, spectral):
        super().__init__()
        self.spectral = spectral
        hidden = 3 * channels
        self.project_in = nn.Conv2d(channels, hidden, 1, bias=False)
        self.before = nn.ModuleList([nn.Conv2d(hidden, hidden, k, padding=k // 2,
                                  groups=hidden, bias=False) for k in (3, 5, 7)])
        self.after = nn.ModuleList([nn.Conv2d(hidden, hidden, k, padding=k // 2,
                                 groups=hidden, bias=False) for k in (3, 5, 7)])
        self.project_out = nn.Conv2d(3 * hidden, channels, 1, bias=False)

    def mix(self, branches):
        values = [torch.fft.rfft2(x) for x in branches] if self.spectral else branches
        chunks = [x.chunk(3, dim=1) for x in values]
        mixed = [torch.cat([chunks[k][part] for k in range(3)], 1) for part in range(3)]
        if self.spectral:
            mixed = [torch.fft.irfft2(x, s=branches[0].shape[-2:]) for x in mixed]
        return mixed

    def forward(self, x):
        projected = self.project_in(x)
        branches = [F.relu(layer(projected)) for layer in self.before]
        mixed = self.mix(branches)
        return self.project_out(torch.cat([F.relu(layer(value)) for layer, value in zip(self.after, mixed)], 1))


def summary(x):
    mean = x.mean((-2, -1), keepdim=True)
    rms = ((x - mean).square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
    return torch.cat((mean, rms), 1)


class FreDFTStage(nn.Module):
    def __init__(self, legacy_stage, spectral):
        super().__init__()
        # Reuse ONLY initialized shell layers. Selector is discarded completely.
        self.local_projection = legacy_stage.local_projection
        self.context_projection = legacy_stage.context_projection
        self.fusion_gate = legacy_stage.fusion_gate
        self.output_projection = legacy_stage.output_projection
        channels = self.local_projection.out_channels
        with torch.random.fork_rng(devices=[]):
            self.mfda = MFDA(channels, spectral)
            self.norm_ffn = ChannelLayerNorm(channels)
            self.fdffl = FDFFL(channels, spectral)

    def forward(self, local, context):
        dtype = local.dtype
        with torch.autocast(device_type=local.device.type, enabled=False):
            local_work = local if dtype == torch.float64 else local.float()
            context_work = F.interpolate(context.to(local_work.dtype), size=local.shape[-2:],
                                         mode='bilinear', align_corners=True)
            left, right = self.mfda(self.local_projection(local_work), self.context_projection(context_work))
            left = left + self.fdffl(self.norm_ffn(left))
            right = right + self.fdffl(self.norm_ffn(right))
            descriptor = torch.cat((summary(left), summary(right)), 1)
            gate = self.fusion_gate(descriptor).reshape(local.shape[0], 2, local.shape[1], 1, 1).softmax(1)
            residual = self.output_projection(gate[:, 0] * left + gate[:, 1] * right)
            output = local_work + residual
            with torch.no_grad():
                weight = gate[:, 1]
                ratio = residual.square().mean().sqrt() / local_work.square().mean().sqrt().clamp_min(1e-6)
                diagnostics = torch.stack((weight.mean(), weight.std(1, unbiased=False).mean(),
                                           weight.std(0, unbiased=False).mean(), ratio))
        return output.to(dtype), diagnostics


class CascadeFreDFT(nn.Module):
    def __init__(self, legacy_cascade, spectral=True):
        super().__init__()
        self.stages = nn.ModuleDict({name: FreDFTStage(block, spectral)
                                    for name, block in legacy_cascade.stages.items()})

    def forward(self, features):
        enhanced = list(features)
        context = features[3]
        diagnostics = {}
        for level in (3, 2, 1):
            context, diagnostics[level] = self.stages[f'p{level}'](features[level - 1], context)
            enhanced[level - 1] = context
        zero = features[0].new_zeros((), dtype=torch.float32)
        fusion = torch.stack([diagnostics[i] for i in (1, 2, 3)])
        # Compatibility with existing decoder output contract; no selector stats.
        return enhanced, zero, features[0].new_empty((0,), dtype=torch.float32), fusion
