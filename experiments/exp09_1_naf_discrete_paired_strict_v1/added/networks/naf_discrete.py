"""Experiment 9: conjugate-paired discrete selection in the experiment-7 cascade.

Training uses sample-keyed Gumbel-Top-K and a biased straight-through sigmoid
relaxation. Evaluation uses stable deterministic Top-K. These are explicit 2D
adaptations, not an assertion about NAF-MSNN's unspecified gradient estimator.
"""
from functools import lru_cache
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


BRANCHES = ('local', 'context')
FILTER_STATS_NAMES = ('keep_low', 'keep_mid', 'keep_high', 'retained_full_fraction',
                      'score_mean', 'score_std', 'normalized_entropy', 'regularizer',
                      'energy_retention', 'relative_change')
FUSION_STATS_NAMES = ('context_weight', 'channel_gate_std', 'image_gate_std', 'residual_ratio')
SELECTION_PROTOCOL = 'naf_discrete_v1'
REGULARIZATION_DENOMINATOR = 6.0


@lru_cache(maxsize=24)
def frequency_geometry(height, width, device, dtype):
    """rFFT indices for independent conjugate pairs; self-conjugate bins stay 1.

    A pair crossing the rFFT boundary is represented once. Boundary-column pairs
    are represented twice but share a score and mask. Group means consequently
    equal full-spectrum means over the selectable (size-two) pairs.
    """
    if min(height, width) < 3:
        raise ValueError('Discrete NAF requires recipient height and width >= 3')
    half = width // 2 + 1
    groups, representatives, partners, mapping = {}, [], [], []
    for y in range(height):
        for x in range(half):
            cy, cx = (-y) % height, (-x) % width
            if (y, x) == (cy, cx):
                mapping.append(-1)
                continue
            key = min(y * width + x, cy * width + cx)
            if key not in groups:
                groups[key] = len(groups)
                representatives.append(y * half + x)
                partners.append(cy * half + cx if cx < half else y * half + x)
            mapping.append(groups[key])
    count = len(groups)
    mapping = [count if index < 0 else index for index in mapping]
    index = torch.tensor(representatives, device=device, dtype=torch.long)
    partner = torch.tensor(partners, device=device, dtype=torch.long)
    mapping = torch.tensor(mapping, device=device, dtype=torch.long)
    fy = (2 * torch.fft.fftfreq(height, device=device, dtype=dtype)).abs()
    fx = 2 * torch.fft.rfftfreq(width, device=device, dtype=dtype)
    y = fy[:, None].expand(height, half)
    x = fx[None, :].expand(height, half)
    radius = (x.square() + y.square()).sqrt() / math.sqrt(2)
    bands = (radius.flatten().index_select(0, index) * 3).long().clamp(max=2)
    anchors = height * width - 2 * count
    return index, partner, mapping, bands, y, x, anchors


class _BudgetedSigmoid(torch.autograd.Function):
    """Smooth K-budget surrogate, with implicit threshold differentiation.

    q=sigmoid((s-t)/tau), sum(q)=K. Its Jacobian is
    diag(d)-d*d.T/sum(d), d=q*(1-q)/tau. This is NOT the derivative
    of hard Top-K and does not make the straight-through estimator unbiased.
    """
    @staticmethod
    def forward(ctx, scores, keep, tau):
        if not 0 < keep < scores.shape[-1] or tau <= 0:
            raise ValueError('Surrogate requires 0 < K < G and tau > 0')
        low = scores.min(-1, keepdim=True).values - 40 * tau
        high = scores.max(-1, keepdim=True).values + 40 * tau
        for _ in range(64 if scores.dtype == torch.float64 else 32):
            threshold = (low + high) * 0.5
            total = ((scores - threshold) / tau).sigmoid().sum(-1, keepdim=True)
            low = torch.where(total > keep, threshold, low)
            high = torch.where(total > keep, high, threshold)
        relaxed = ((scores - (low + high) * 0.5) / tau).sigmoid()
        ctx.save_for_backward(relaxed)
        ctx.tau = tau
        return relaxed

    @staticmethod
    def backward(ctx, gradient):
        (q,) = ctx.saved_tensors
        d = q * (1 - q) / ctx.tau
        denominator = d.sum(-1, keepdim=True).clamp_min(torch.finfo(d.dtype).tiny)
        return d * (gradient - (gradient * d).sum(-1, keepdim=True) / denominator), None, None


def discrete_mask(scores, keep, training, noise=None, tau=0.1):
    """Exactly K ones forward; budget-conserving surrogate backward in training."""
    if not 0 < keep <= scores.shape[-1]:
        raise ValueError('K must be between 1 and G')
    if keep == scores.shape[-1]:
        return scores * 0 + 1
    if training:
        if noise is None or noise.shape != scores.shape:
            raise ValueError('Training selection requires sample-keyed Gumbel noise')
        scores = scores + noise
    # Stable order fixes ties without adding a frequency-dependent score bias.
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :keep]
    hard = torch.zeros_like(scores).scatter(-1, order, 1)
    if not training:
        return hard
    soft = _BudgetedSigmoid.apply(scores, keep, tau)
    return hard + (soft - soft.detach())


def sample_gumbels(seed, epoch, ids, level, branch, count, device, dtype):
    """CPU-local generators: no consumption of model, input or CUDA RNG streams."""
    rows = []
    for sample_id in ids:
        key = json.dumps([SELECTION_PROTOCOL, int(seed), int(epoch), str(sample_id),
                          int(level), branch], ensure_ascii=False, separators=(',', ':')).encode()
        sample_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') & ((1 << 63) - 1)
        generator = torch.Generator(device='cpu').manual_seed(sample_seed)
        uniform = torch.rand(count, generator=generator, device='cpu', dtype=torch.float64)
        uniform.clamp_(min=torch.finfo(uniform.dtype).eps, max=1 - torch.finfo(uniform.dtype).eps)
        rows.append(-(-uniform.log()).log())
    return torch.stack(rows).to(device=device, dtype=dtype)


def spatial_summary(features):
    mean = features.mean((-2, -1), keepdim=True)
    rms = ((features - mean).square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
    return torch.cat([mean, rms], dim=1)


class DualInputDiscreteNAF2D(nn.Module):
    keep = 0.9
    relaxation_tau = 0.1  # Gradient surrogate only; selection softmax temperature is 1.

    def __init__(self, level, channels=64):
        super().__init__()
        if level not in (1, 2, 3):
            raise ValueError('Only NAF1, NAF2 and NAF3 exist')
        self.level = level
        self.selection_context = None
        with torch.random.fork_rng(devices=[]):
            # Same construction order/shapes as experiment 7, preserving every
            # shared projection/gate initial tensor and the outer CPU RNG state.
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
            nn.init.zeros_(self.output_projection.weight)

    def set_selection_context(self, seed, epoch, image_ids):
        self.selection_context = (int(seed), int(epoch), tuple(str(x) for x in image_ids))

    def selection(self, local_spectrum, context_spectrum, spatial_size):
        height, width = spatial_size
        index, partner, mapping, bands, y, x, anchors = frequency_geometry(
            height, width, local_spectrum.device, local_spectrum.real.dtype)
        count = index.numel()

        def describe(spectrum):
            magnitude = spectrum.abs().mean(dim=1, keepdim=True)
            # Exclude DC only from magnitude normalization, as in experiment 7.
            weights = torch.full_like(magnitude[0:1], 2)
            weights[..., 0] = 1
            if width % 2 == 0:
                weights[..., -1] = 1
            weights[..., 0, 0] = 0
            scale = (magnitude * weights).sum((-2, -1), keepdim=True) / weights.sum()
            return torch.log1p(magnitude / scale.clamp_min(1e-6))

        local, context = describe(local_spectrum), describe(context_spectrum)
        descriptor = torch.cat([local, context, (local - context).abs(),
                                y[None, None].expand_as(local), x[None, None].expand_as(local)], dim=1)
        values = self.selector(descriptor).sigmoid().flatten(2)
        values = (values.index_select(-1, index) + values.index_select(-1, partner)) * 0.5
        log_probability = values.log_softmax(-1)
        keep = math.ceil(self.keep * count)
        noise = None
        if self.training:
            if self.selection_context is None or len(self.selection_context[2]) != values.shape[0]:
                raise RuntimeError('Set seed/epoch/sample IDs before each training forward')
            seed, epoch, ids = self.selection_context
            noise = torch.stack([sample_gumbels(seed, epoch, ids, self.level, branch,
                                count, values.device, values.dtype) for branch in BRANCHES], dim=1)
        masks = discrete_mask(log_probability, keep, self.training, noise, self.relaxation_tau)
        # Expand the same group mask to both represented partners; anchors=1.
        flat = torch.cat([masks, torch.ones_like(masks[..., :1])], dim=-1).index_select(-1, mapping)
        gates = flat.reshape(values.shape[0], 2, height, width // 2 + 1)
        scales = values.new_tensor([self.level, self.level + 1]).view(1, 2, 1)
        coefficients = scales ** (-(bands.to(values.dtype)[None, None] + 1))
        penalties = (values.square() * coefficients).mean((0, 2))
        return gates, masks, values, penalties, bands, anchors, log_probability

    def filter_pair(self, local, context):
        spectra = [torch.fft.rfft2(t, norm='ortho') for t in (local, context)]
        gates, masks, values, penalties, bands, anchors, log_p = self.selection(*spectra, local.shape[-2:])
        filtered = [torch.fft.irfft2(q * gates[:, b:b + 1], s=local.shape[-2:], norm='ortho')
                    for b, q in enumerate(spectra)]
        with torch.no_grad():
            means = [(masks * (bands == band)).sum((0, 2)) /
                     ((bands == band).sum().clamp_min(1) * masks.shape[0]) for band in range(3)]
            full_fraction = (2 * masks.sum(-1) + anchors) / (local.shape[-2] * local.shape[-1])
            energy, change = [], []
            for original, restored in zip((local, context), filtered):
                denominator = original.square().sum((1, 2, 3)).clamp_min(1e-12)
                energy.append((restored.square().sum((1, 2, 3)) / denominator).mean())
                change.append(((restored - original).square().sum((1, 2, 3)) / denominator).sqrt().mean())
            entropy = -(log_p.exp() * log_p).sum(-1) / math.log(values.shape[-1])
            stats = torch.stack([*means, full_fraction.mean(0), values.mean((0, 2)),
                values.std(-1, unbiased=False).mean(0), entropy.mean(0), penalties,
                torch.stack(energy), torch.stack(change)], dim=-1)
        return filtered[0], filtered[1], penalties, stats

    def forward(self, local, context):
        original_dtype = local.dtype
        with torch.autocast(device_type=local.device.type, enabled=False):
            work = local if original_dtype == torch.float64 else local.float()
            coarse = F.interpolate(context.to(work.dtype), size=local.shape[-2:],
                                   mode='bilinear', align_corners=True)
            local_filtered, context_filtered, penalties, stats = self.filter_pair(
                self.local_projection(work), self.context_projection(coarse))
            descriptor = torch.cat([spatial_summary(local_filtered), spatial_summary(context_filtered)], dim=1)
            gate = self.fusion_gate(descriptor).reshape(work.shape[0], 2, work.shape[1], 1, 1).softmax(1)
            residual = self.output_projection(gate[:, 0] * local_filtered + gate[:, 1] * context_filtered)
            output = work + residual
            with torch.no_grad():
                context_weight = gate[:, 1]
                ratio = residual.square().mean().sqrt() / work.square().mean().sqrt().clamp_min(1e-6)
                fusion = torch.stack([context_weight.mean(), context_weight.std(1, unbiased=False).mean(),
                                      context_weight.std(0, unbiased=False).mean(), ratio])
        return output.to(original_dtype), penalties, stats, fusion


class CascadeDiscreteNAF2D(nn.Module):
    """F3=NAF3(P3,P4); F2=NAF2(P2,F3); F1=NAF1(P1,F2); no NAF4."""
    def __init__(self, channels=64):
        super().__init__()
        self.stages = nn.ModuleDict({'p%d' % i: DualInputDiscreteNAF2D(i, channels) for i in (1, 2, 3)})

    def set_selection_context(self, seed, epoch, image_ids):
        for block in self.stages.values():
            block.set_selection_context(seed, epoch, image_ids)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError('Expected four RFB outputs P1,P2,P3,P4')
        enhanced = list(features)
        context = features[3]
        penalties, statistics, fusion = {}, {}, {}
        for level in (3, 2, 1):
            context, penalty, stats, gate = self.stages['p%d' % level](features[level - 1], context)
            enhanced[level - 1] = context
            penalties[level], statistics[level], fusion[level] = penalty, stats, gate
        regularizer = torch.stack([penalties[i] for i in (1, 2, 3)]).sum() / REGULARIZATION_DENOMINATOR
        return (enhanced, regularizer, torch.stack([statistics[i] for i in (1, 2, 3)]),
                torch.stack([fusion[i] for i in (1, 2, 3)]))
