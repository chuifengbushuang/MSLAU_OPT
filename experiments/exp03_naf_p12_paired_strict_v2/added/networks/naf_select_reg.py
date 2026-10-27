"""NAF v1: budgeted 2-D selection shared by the full model and stage ablations.

Adaptation of NAF-MSNN, not a reproduction of its time-series sampling/fusion.
No dependencies beyond torch. FFT and selection use FP32 under autocast.
"""
import math

import torch
from torch import nn


REGULARIZATION_DENOMINATOR = 4.0


STATS_NAMES = ('keep_low', 'keep_mid', 'keep_high', 'alpha', 'regularizer',
               'selection_std', 'keep_all')


def soft_budget_mask(values, multiplicity, keep=0.9, temperature=0.1):
    """Capped probability allocation: g=min(1,c*p), sum(w*g)=keep*sum(w).

    w counts the full real spectrum represented by rfft2 (DC weight is zero).
    Sort once to find the saturated set; solve c analytically on that set.
    The selected branch is differentiable almost everywhere, including through c.
    """
    shape = values.shape
    logits = values.flatten(1) / temperature
    weights = multiplicity.expand_as(values).flatten(1)
    density = torch.softmax(logits, dim=-1)
    # Normalization across the represented full spectrum. The common factor
    # cancels in c*p; retaining it makes the selection probability explicit.
    density = density / (density * weights).sum(-1, keepdim=True)
    ordered, order = density.sort(dim=-1, descending=True)
    ordered_weights = weights.gather(-1, order)
    prefix = torch.cat([weights.new_zeros((weights.shape[0], 1)),
                        ordered_weights.cumsum(-1)[:, :-1]], dim=-1)
    suffix = (ordered * ordered_weights).flip(-1).cumsum(-1).flip(-1)
    budget = keep * weights.sum(-1, keepdim=True)
    # A suffix containing only excluded DC has zero mass; keep that candidate
    # finite and ineligible rather than producing infinities in unused branches.
    scales = (budget - prefix) / torch.where(suffix > 0, suffix, torch.ones_like(suffix))
    valid = (suffix > 0) & (scales >= 0) & (scales * ordered <= 1)
    index = valid.to(torch.int64).argmax(-1, keepdim=True)
    scale = scales.gather(-1, index)
    gate = (scale * density).clamp(max=1).reshape(shape)
    # DC is explicitly preserved and excluded from the selection budget.
    return torch.where(multiplicity > 0, gate, torch.ones_like(gate))


def frequency_geometry(height, width, device, dtype):
    fy = (2 * torch.fft.fftfreq(height, device=device, dtype=dtype)).abs()
    fx = 2 * torch.fft.rfftfreq(width, device=device, dtype=dtype)
    y = fy.view(1, 1, height, 1).expand(1, 1, height, fx.numel())
    x = fx.view(1, 1, 1, -1).expand_as(y)
    radius = (x.square() + y.square()).sqrt() / math.sqrt(2)
    bands = (radius * 3).long().clamp(max=2)
    columns = torch.full_like(fx, 2)
    columns[0] = 1
    if width % 2 == 0:
        columns[-1] = 1
    weights = columns.view(1, 1, 1, -1).expand_as(y).clone()
    weights[..., 0, 0] = 0
    return y, x, bands, weights


class BudgetNAF2D(nn.Module):
    """Per-image, channel-shared frequency selector for one existing feature map."""
    keep = 0.9
    temperature = 0.1
    max_alpha = 0.25

    def __init__(self, scale_index):
        super().__init__()
        self.scale_index = scale_index
        # Isolate added initialization so baseline decoder weights AND the global
        # RNG stream for subsequent training remain unaffected by this addition.
        with torch.random.fork_rng(devices=[]):
            self.selector = nn.Sequential(nn.Conv2d(3, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1))
            nn.init.zeros_(self.selector[-1].weight)
            nn.init.zeros_(self.selector[-1].bias)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(0.1 / (self.max_alpha - 0.1))))

    def selection(self, spectrum, spatial_size):
        height, width = spatial_size
        y, x, bands, weights = frequency_geometry(height, width, spectrum.device, spectrum.real.dtype)
        magnitude = spectrum.abs().mean(dim=1, keepdim=True)
        scale = (magnitude * weights).sum((-2, -1), keepdim=True) / weights.sum()
        descriptor = torch.log1p(magnitude / scale.clamp_min(1e-6))
        descriptor = torch.cat([descriptor, y.expand_as(descriptor), x.expand_as(descriptor)], dim=1)
        # Conv is explicitly FP32 in normal/mixed-precision training.
        values = self.selector(descriptor).sigmoid()
        # On kx=0 (and Nyquist for even width), rfft stores both conjugate ky
        # entries. Their real gates must match for a valid real inverse transform.
        reverse = (-torch.arange(height, device=spectrum.device)) % height
        paired = values.index_select(-2, reverse)
        boundary = torch.zeros_like(weights, dtype=torch.bool)
        boundary[..., 0] = True
        if width % 2 == 0:
            boundary[..., -1] = True
        values = torch.where(boundary, (values + paired) * 0.5, values)
        gate = soft_budget_mask(values, weights, self.keep, self.temperature)
        return gate, bands, weights

    def forward(self, features):
        original_dtype = features.dtype
        # torch.fft on 11/22-sized maps must not use half-precision cuFFT.
        with torch.autocast(device_type=features.device.type, enabled=False):
            work = features if original_dtype == torch.float64 else features.float()
            spectrum = torch.fft.rfft2(work, norm='ortho')
            gate, bands, weights = self.selection(spectrum, features.shape[-2:])
            reconstructed = torch.fft.irfft2(spectrum * gate, s=features.shape[-2:], norm='ortho')
            alpha = self.max_alpha * self.alpha_logit.sigmoid()
            output = work + alpha * (reconstructed - work)
            # Apply the paper's n^-1 / n^-2 / n^-3 profile to the ACTUAL
            # budgeted retention mask. This prevents shrinking all pre-softmax
            # scores to evade a penalty without changing frequency selection.
            coefficients = self.scale_index ** (-(bands.to(work.dtype) + 1))
            penalty = (gate.square() * coefficients * weights).sum() / (weights.sum() * features.shape[0])
            band_means = []
            for band in range(3):
                band_weights = weights * (bands == band)
                band_means.append((gate * band_weights).sum() /
                                  (band_weights.sum().clamp_min(1) * features.shape[0]))
            mean = (gate * weights).sum() / (weights.sum() * features.shape[0])
            std = (((gate - mean).square() * weights).sum() /
                   (weights.sum() * features.shape[0])).sqrt()
            # Diagnostics must not keep a graph or alter the regularizer gradient.
            stats = torch.stack([*band_means, alpha, penalty, std, mean]).detach()
        return output.to(original_dtype), penalty, stats


def parse_stages(value):
    """Canonical anatomical indices; disabling stages never renumbers them."""
    stages = tuple(int(part) for part in value.split(',')) if isinstance(value, str) else tuple(value)
    if not stages or len(set(stages)) != len(stages) or any(stage not in (1, 2, 3, 4) for stage in stages):
        raise ValueError('NAF stages must be distinct indices from 1,2,3,4')
    return tuple(sorted(stages))


class PyramidNAF2D(nn.Module):
    """Independent selectors at configured levels; no cross-level fusion."""

    def __init__(self, active_stages=(1, 2, 3, 4)):
        super().__init__()
        self.active_stages = parse_stages(active_stages)
        self.stages = nn.ModuleDict({
            'p%d' % stage: BudgetNAF2D(scale_index=stage) for stage in self.active_stages
        })

    def forward(self, features):
        enhanced = list(features)
        penalties, diagnostics = [], []
        for stage in self.active_stages:
            enhanced[stage - 1], penalty, stats = self.stages['p%d' % stage](features[stage - 1])
            penalties.append(penalty)
            diagnostics.append(stats)
        # Fixed anatomical denominator, NOT mean over active modules. Removing
        # one module removes its term without increasing other modules' weights.
        regularizer = torch.stack(penalties).sum() / REGULARIZATION_DENOMINATOR
        return enhanced, regularizer, torch.stack(diagnostics)
