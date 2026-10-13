"""NAF-inspired 2D adaptation, not a reproduction of an official implementation.

See docs/frequency_enhancement/method_design.md for the paper/engineering split.
No cross-stage exchange and no mutable forward-result caches are used here.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def frequency_grid(height, width, *, device, dtype):
    """Coordinates in actual (unshifted) rfft2 order, normalized to Nyquist."""
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype)[:, None]
    fx = torch.fft.rfftfreq(width, device=device, dtype=dtype)[None, :]
    yy, xx = torch.broadcast_tensors(2 * fy.abs(), 2 * fx)
    radius = torch.sqrt((yy.square() + xx.square()) / 2)
    multiplicity = torch.full_like(radius, 2)
    multiplicity[:, 0] = 1
    if width % 2 == 0:
        multiplicity[:, -1] = 1
    return yy, xx, radius, multiplicity


def symmetrize_special_columns(mask, width):
    """Keep the DC/Nyquist columns compatible with a real spatial signal."""
    height = mask.shape[-2]
    reverse = (-torch.arange(height, device=mask.device)) % height
    special = torch.zeros(mask.shape[-1], dtype=torch.bool, device=mask.device)
    special[0] = True
    if width % 2 == 0:
        special[-1] = True
    return torch.where(special, (mask + mask.index_select(-2, reverse)) * 0.5, mask)


class AdaptiveFrequencyEnhancement2D(nn.Module):
    """Shape/dtype preserving, input-conditioned frequency enhancement.

    forward(x) returns only the tensor. forward_with_details returns explicitly
    requested auxiliary information; only the regularization scalar keeps a graph.
    Keep parameters in FP32 for ordinary AMP training (model.half also works).
    """

    def __init__(self, channels=64, hidden_channels=16, band_edges=(1/3, 2/3),
                 band_weights=(1., 1., 1.), alpha_init=0.1):
        super().__init__()
        if channels < 1 or hidden_channels < 1:
            raise ValueError("channels and hidden_channels must be positive")
        if len(band_edges) != 2 or not 0 < band_edges[0] < band_edges[1] < 1:
            raise ValueError("band_edges must satisfy 0 < low < high < 1")
        if (len(band_weights) != 3 or
                any(not math.isfinite(v) or v < 0 for v in band_weights) or
                sum(band_weights) <= 0):
            raise ValueError("three finite nonnegative band weights with positive sum required")
        if not 0 < alpha_init < 1:
            raise ValueError("alpha_init must be strictly between 0 and 1")
        self.channels = channels
        self.band_edges = tuple(band_edges)
        self.band_weights = tuple(band_weights)
        self.gate_in = nn.Conv2d(channels + 3, hidden_channels, 1)
        self.gate_out = nn.Conv2d(hidden_channels, channels, 1)
        nn.init.normal_(self.gate_out.weight, std=0.02)
        nn.init.constant_(self.gate_out.bias, 0.01)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1-alpha_init))))

    @property
    def alpha(self):
        return self.alpha_logit.sigmoid()

    def _validate(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels or min(x.shape) < 1:
            raise ValueError("expected nonempty [B,{},H,W] input".format(self.channels))
        if not x.is_floating_point():
            raise TypeError("AFFE requires real floating-point input")

    def _frequency_weights(self, spectrum, width, yy, xx):
        magnitude = torch.log1p(spectrum.abs())
        batch, _, height, half_width = magnitude.shape
        context = magnitude.mean(dim=(1, 2, 3), keepdim=True).expand(batch, 1, height, half_width)
        coordinates = torch.stack((yy, xx))[None].expand(batch, -1, -1, -1)
        gate_input = torch.cat((magnitude, context, coordinates), dim=1)
        dtype = magnitude.dtype
        hidden = F.silu(F.conv2d(gate_input, self.gate_in.weight.to(dtype), self.gate_in.bias.to(dtype)))
        logits = F.conv2d(hidden, self.gate_out.weight.to(dtype), self.gate_out.bias.to(dtype))
        mask = 1 + 0.5 * logits.tanh()
        return symmetrize_special_columns(mask, width)

    def frequency_weights(self, x):
        """Return differentiable real [B,C,H,W//2+1] weights, without caching."""
        self._validate(x)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            xx = x.to(dtype)
            spectrum = torch.fft.rfft2(xx, dim=(-2, -1), norm="ortho")
            yy, fx, _, _ = frequency_grid(*x.shape[-2:], device=x.device, dtype=dtype)
            return self._frequency_weights(spectrum, x.shape[-1], yy, fx)

    def _band_statistics(self, mask, radius, multiplicity):
        low, high = self.band_edges
        bands = (radius < low, (radius >= low) & (radius < high), radius >= high)
        penalties, means, counts, active_weights = [], [], [], []
        for band, weight in zip(bands, self.band_weights):
            measure = multiplicity * band
            count = measure.sum()
            denominator = count.clamp_min(1) * mask.shape[0] * mask.shape[1]
            penalties.append(((mask - 1).square() * measure).sum() / denominator)
            means.append((mask * measure).sum() / denominator)
            counts.append(count)
            active_weights.append((count > 0).to(mask.dtype) * weight)
        weights = torch.stack(active_weights)
        regularization = (torch.stack(penalties) * weights).sum() / weights.sum().clamp_min(1e-12)
        return regularization, torch.stack(means), torch.stack(counts)

    def forward_with_details(self, x, *, return_regularization=False, return_diagnostics=False):
        self._validate(x)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            spatial = x.to(dtype)
            spectrum = torch.fft.rfft2(spatial, dim=(-2, -1), norm="ortho")
            yy, xx, radius, multiplicity = frequency_grid(*x.shape[-2:], device=x.device, dtype=dtype)
            mask = self._frequency_weights(spectrum, x.shape[-1], yy, xx)
            filtered_spectrum = spectrum * mask
            filtered = torch.fft.irfft2(filtered_spectrum, s=x.shape[-2:], dim=(-2, -1), norm="ortho")
            alpha = self.alpha_logit.to(dtype).sigmoid()
            enhanced = spatial + alpha * (filtered - spatial)
            reg = spatial.new_zeros(())
            diagnostics = None
            if return_regularization or return_diagnostics:
                reg, means, counts = self._band_statistics(mask, radius, multiplicity)
            if return_diagnostics:
                with torch.no_grad():
                    # Full-spectrum-equivalent relative L2 change (half-spectrum weights).
                    energy = (spectrum.abs().square() * multiplicity).sum()
                    change = ((filtered_spectrum-spectrum).abs().square() * multiplicity).sum()
                    diagnostics = {
                        "weights": mask.detach(),
                        "band_mean_weights": means.detach(),
                        "band_counts": counts.detach(),
                        "input_norm": spatial.detach().norm(),
                        "filtered_norm": filtered.detach().norm(),
                        "output_norm": enhanced.detach().norm(),
                        "alpha": alpha.detach(),
                        "regularization": reg.detach(),
                        "relative_spectrum_change": (change / energy.clamp_min(1e-30)).sqrt(),
                    }
        return enhanced.to(x.dtype), reg if return_regularization else None, diagnostics

    def forward(self, x):
        return self.forward_with_details(x)[0]


class FourStageFrequencyEnhancer(nn.Module):
    """Four independent modules. Disabled stages return the identical input object."""

    def __init__(self, channels=64, enabled_stages=(True, True, True, True), **kwargs):
        super().__init__()
        if len(enabled_stages) != 4 or any(type(v) is not bool for v in enabled_stages):
            raise ValueError("enabled_stages requires exactly four booleans")
        self.enabled_stages = tuple(enabled_stages)
        self.stages = nn.ModuleList([
            AdaptiveFrequencyEnhancement2D(channels, **kwargs) for _ in range(4)
        ])

    def enhance_features(self, features, *, return_regularization=False, return_diagnostics=False):
        if len(features) != 4:
            raise ValueError("exactly four ordered features are required")
        outputs, regularizations, diagnostics = [], [], []
        for feature, stage, enabled in zip(features, self.stages, self.enabled_stages):
            stage._validate(feature)
            if enabled:
                output, reg, diag = stage.forward_with_details(
                    feature, return_regularization=return_regularization,
                    return_diagnostics=return_diagnostics)
                if return_regularization:
                    regularizations.append(reg)
            else:
                output, diag = feature, None
            outputs.append(output)
            if return_diagnostics:
                diagnostics.append(diag)
        if not (return_regularization or return_diagnostics):
            return outputs
        reg = (torch.stack(regularizations).mean() if regularizations
               else features[0].new_zeros((), dtype=torch.float32))
        return outputs, reg if return_regularization else None, diagnostics if return_diagnostics else None

    def forward(self, features):
        return self.enhance_features(features)


def load_frequency_state_dict(model, state_dict, *, allow_legacy=False):
    """Strict loading, or an audited old-P5.1 -> AFFE migration.

    Validate all keys and shapes BEFORE copying. Legacy migration permits only
    the complete set of new AFFE keys to be absent; partial/corrupt AFFE states,
    missing old parameters, and unexpected keys are errors.
    """
    expected = model.state_dict()
    missing = set(expected) - set(state_dict)
    unexpected = set(state_dict) - set(expected)
    new_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, FourStageFrequencyEnhancer):
            prefix = name + "." if name else ""
            new_keys.update(prefix + key for key in module.state_dict())
    bad_shapes = [key for key in set(expected) & set(state_dict)
                  if expected[key].shape != state_dict[key].shape]
    legacy = allow_legacy and bool(new_keys) and missing == new_keys
    if unexpected or bad_shapes or (missing and not legacy):
        raise RuntimeError("Checkpoint audit failed: missing={}, unexpected={}, shape_mismatch={}".format(
            sorted(missing), sorted(unexpected), sorted(bad_shapes)))
    result = model.load_state_dict(state_dict, strict=not legacy)
    if set(result.missing_keys) != missing or result.unexpected_keys:
        raise RuntimeError("Checkpoint load disagrees with preflight audit")
    return {"missing_keys": sorted(missing), "unexpected_keys": sorted(unexpected),
            "loaded_keys": len(state_dict), "legacy_migration": legacy}
