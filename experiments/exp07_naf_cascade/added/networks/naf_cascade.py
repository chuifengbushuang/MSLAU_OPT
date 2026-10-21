"""Experiment 7: three residual, dual-input frequency-selection/fusion blocks.

Uses SPLF-SAM's adjacent-level/residual fusion topology and NAF-MSNN's
filter-then-exchange principle. Neither paper is reproduced verbatim. This
standalone file intentionally does not import the frozen NAF v1 experiment.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


REGULARIZATION_DENOMINATOR = 6.0
BRANCHES = ('local', 'context')
FILTER_STATS_NAMES = ('keep_low', 'keep_mid', 'keep_high', 'alpha', 'regularizer',
                      'selection_std', 'keep_all', 'image_selection_std')
FUSION_STATS_NAMES = ('context_weight', 'channel_gate_std', 'image_gate_std',
                      'residual_ratio')


def soft_budget_mask(values, multiplicity, keep=0.9, temperature=0.1):
    """Allocate a separate full-spectrum-weighted soft budget to EACH row.

    Same capped allocation as frozen v1: g=min(1,c*p). The caller flattens
    batch AND branch into rows, so the two branches never share a budget.
    Sorting selects a differentiable analytic solution, not a hard top-k mask.
    """
    shape = values.shape
    logits = values.flatten(1) / temperature
    weights = multiplicity.expand_as(values).flatten(1)
    density = torch.softmax(logits, dim=-1)
    density = density / (density * weights).sum(-1, keepdim=True)
    ordered, order = density.sort(dim=-1, descending=True)
    ordered_weights = weights.gather(-1, order)
    prefix = torch.cat([weights.new_zeros((weights.shape[0], 1)),
                        ordered_weights.cumsum(-1)[:, :-1]], dim=-1)
    suffix = (ordered * ordered_weights).flip(-1).cumsum(-1).flip(-1)
    budget = keep * weights.sum(-1, keepdim=True)
    scales = (budget - prefix) / torch.where(suffix > 0, suffix, torch.ones_like(suffix))
    valid = (suffix > 0) & (scales >= 0) & (scales * ordered <= 1)
    index = valid.to(torch.int64).argmax(-1, keepdim=True)
    gate = (scales.gather(-1, index) * density).clamp(max=1).reshape(shape)
    return torch.where(multiplicity > 0, gate, torch.ones_like(gate))


def frequency_geometry(height, width, device, dtype):
    """rFFT multiplicities, non-DC support and radial thirds of Nyquist square."""
    if height < 2 or width < 2:
        raise ValueError('NAF recipient features must have height and width >= 2')
    fy = (2 * torch.fft.fftfreq(height, device=device, dtype=dtype)).abs()
    fx = 2 * torch.fft.rfftfreq(width, device=device, dtype=dtype)
    y = fy.view(1, 1, height, 1).expand(1, 1, height, fx.numel())
    x = fx.view(1, 1, 1, -1).expand_as(y)
    bands = (((x.square() + y.square()).sqrt() / math.sqrt(2)) * 3).long().clamp(max=2)
    columns = torch.full_like(fx, 2)
    columns[0] = 1
    if width % 2 == 0:
        columns[-1] = 1
    weights = columns.view(1, 1, 1, -1).expand_as(y).clone()
    weights[..., 0, 0] = 0
    return y, x, bands, weights


def spatial_summary(features):
    """Mean AND centered RMS: a DC-preserving filter cannot change GAP alone."""
    mean = features.mean((-2, -1), keepdim=True)
    rms = ((features - mean).square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
    return torch.cat([mean, rms], dim=1)


class DualInputNAF2D(nn.Module):
    keep = 0.9
    temperature = 0.1
    max_alpha = 0.25

    def __init__(self, level, channels=64):
        super().__init__()
        if level not in (1, 2, 3):
            raise ValueError('Only NAF1, NAF2 and NAF3 exist in experiment 7')
        self.level = level
        # All added random initialization is isolated from the original RFB,
        # decoder and subsequent training RNG. No CUDA generator is touched.
        with torch.random.fork_rng(devices=[]):
            self.local_projection = nn.Conv2d(channels, channels, 1, bias=False)
            self.context_projection = nn.Conv2d(channels, channels, 1, bias=False)
            self.selector = nn.Sequential(nn.Conv2d(5, 16, 1), nn.GELU(), nn.Conv2d(16, 2, 1))
            self.fusion_gate = nn.Sequential(nn.Conv2d(4 * channels, 16, 1), nn.GELU(),
                                             nn.Conv2d(16, 2 * channels, 1))
            self.output_projection = nn.Conv2d(channels, channels, 1, bias=False)
            nn.init.zeros_(self.selector[-1].weight)
            nn.init.zeros_(self.selector[-1].bias)
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.zeros_(self.fusion_gate[-1].bias)
            # An exact Pi shortcut, not v1's initially 0.99-scaled non-DC signal.
            nn.init.zeros_(self.output_projection.weight)
        initial = math.log(0.1 / (self.max_alpha - 0.1))
        self.alpha_logit = nn.Parameter(torch.full((2,), initial))

    def selection(self, local_spectrum, context_spectrum, spatial_size):
        """Two masks conditioned jointly on the two normalized magnitude maps."""
        height, width = spatial_size
        y, x, bands, weights = frequency_geometry(
            height, width, local_spectrum.device, local_spectrum.real.dtype)

        def describe(spectrum):
            magnitude = spectrum.abs().mean(dim=1, keepdim=True)
            scale = (magnitude * weights).sum((-2, -1), keepdim=True) / weights.sum()
            return torch.log1p(magnitude / scale.clamp_min(1e-6))

        local, context = describe(local_spectrum), describe(context_spectrum)
        descriptor = torch.cat([local, context, (local - context).abs(),
                                y.expand_as(local), x.expand_as(local)], dim=1)
        values = self.selector(descriptor).sigmoid()
        # kx=0/Nyquist contain represented conjugate ky pairs. An independent
        # real gate on each member would break real-spectrum consistency.
        reverse = (-torch.arange(height, device=values.device)).remainder(height)
        boundary = torch.zeros_like(weights, dtype=torch.bool)
        boundary[..., 0] = True
        if width % 2 == 0:
            boundary[..., -1] = True
        values = torch.where(boundary, (values + values.index_select(-2, reverse)) * 0.5, values)
        batch = values.shape[0]
        gates = soft_budget_mask(values.reshape(batch * 2, 1, height, width // 2 + 1),
                                 weights, self.keep, self.temperature)
        return gates.reshape_as(values), bands, weights

    def filter_pair(self, local, context):
        """Aligned/projected FP32 inputs -> independent phase-preserving filters."""
        local_spectrum = torch.fft.rfft2(local, norm='ortho')
        context_spectrum = torch.fft.rfft2(context, norm='ortho')
        gates, bands, weights = self.selection(local_spectrum, context_spectrum, local.shape[-2:])
        alpha = self.max_alpha * self.alpha_logit.sigmoid()
        filtered = []
        for branch, (features, spectrum) in enumerate(((local, local_spectrum), (context, context_spectrum))):
            restored = torch.fft.irfft2(spectrum * gates[:, branch:branch + 1],
                                        s=local.shape[-2:], norm='ortho')
            filtered.append(features + alpha[branch] * (restored - features))
        # Labels refer to input hierarchy, NOT the order of the three modules.
        # A cascaded context mixes levels; n=level+1 is an explicit prior only.
        scales = gates.new_tensor([self.level, self.level + 1]).view(1, 2, 1, 1)
        coefficients = scales ** (-(bands.to(gates.dtype) + 1))
        penalties = (gates.square() * coefficients * weights).sum((0, 2, 3)) / (
            weights.sum() * gates.shape[0])
        with torch.no_grad():
            means = [(gates * weights * (bands == band)).sum((0, 2, 3)) /
                     ((weights * (bands == band)).sum().clamp_min(1) * gates.shape[0])
                     for band in range(3)]
            per_image_mean = (gates * weights).sum((-2, -1), keepdim=True) / weights.sum()
            frequency_std = (((gates - per_image_mean).square() * weights).sum((0, 2, 3)) /
                             (weights.sum() * gates.shape[0])).sqrt()
            image_std = ((gates.var(dim=0, unbiased=False) * weights[0]).sum((-2, -1)) /
                         weights.sum()).sqrt()
            stats = torch.stack([*means, alpha, penalties, frequency_std,
                                 per_image_mean.mean((0, 2, 3)), image_std], dim=-1)
        return filtered[0], filtered[1], penalties, stats

    def forward(self, local, context):
        original_dtype = local.dtype
        # Full block uses FP32 under autocast; supports double for numeric tests.
        with torch.autocast(device_type=local.device.type, enabled=False):
            work = local if original_dtype == torch.float64 else local.float()
            coarse = context.to(work.dtype)
            coarse = F.interpolate(coarse, size=local.shape[-2:], mode='bilinear', align_corners=True)
            projected_local = self.local_projection(work)
            projected_context = self.context_projection(coarse)
            filtered_local, filtered_context, penalties, filter_stats = self.filter_pair(
                projected_local, projected_context)
            descriptor = torch.cat([spatial_summary(filtered_local), spatial_summary(filtered_context)], dim=1)
            logits = self.fusion_gate(descriptor).reshape(local.shape[0], 2, local.shape[1], 1, 1)
            gate = logits.softmax(dim=1)
            fused = gate[:, 0] * filtered_local + gate[:, 1] * filtered_context
            residual = self.output_projection(fused)
            output = work + residual
            with torch.no_grad():
                context_weight = gate[:, 1]
                ratio = residual.square().mean().sqrt() / work.square().mean().sqrt().clamp_min(1e-6)
                fusion_stats = torch.stack([context_weight.mean(),
                    context_weight.std(dim=1, unbiased=False).mean(),
                    context_weight.std(dim=0, unbiased=False).mean(), ratio])
        return output.to(original_dtype), penalties, filter_stats, fusion_stats


class CascadeNAF2D(nn.Module):
    """F3=NAF3(P3,P4); F2=NAF2(P2,F3); F1=NAF1(P1,F2). No NAF4."""

    def __init__(self, channels=64):
        super().__init__()
        self.stages = nn.ModuleDict({
            'p%d' % level: DualInputNAF2D(level, channels) for level in (1, 2, 3)
        })

    def forward(self, features):
        if len(features) != 4:
            raise ValueError('Expected the four original RFB outputs P1,P2,P3,P4')
        enhanced = list(features)
        context = features[3]
        penalties, filter_stats, fusion_stats = {}, {}, {}
        for level in (3, 2, 1):
            context, penalty, stats, fusion = self.stages['p%d' % level](features[level - 1], context)
            enhanced[level - 1] = context
            penalties[level], filter_stats[level], fusion_stats[level] = penalty, stats, fusion
        # B/frequency means inside R_i,branch; fixed sum of SIX branch penalties.
        regularizer = torch.stack([penalties[i] for i in (1, 2, 3)]).sum() / REGULARIZATION_DENOMINATOR
        return (enhanced, regularizer, torch.stack([filter_stats[i] for i in (1, 2, 3)]),
                torch.stack([fusion_stats[i] for i in (1, 2, 3)]))
