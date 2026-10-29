"""Autograd DCNv4 sampling, transcribed from the pinned author's CUDA kernel.

The original module's projections, grouping, offsets and unnormalized masks are
retained. This backend uses gather for deterministic CUDA backward. It is not a
normal convolution substitute. See tests against an independent grid_sample
oracle, finite differences, padding, non-unit offset scales and center removal.
"""
import torch


def dcnv4_reference(value, offset_mask, kernel_h, kernel_w, stride_h, stride_w,
                    pad_h, pad_w, dilation_h, dilation_w, group, group_channels,
                    offset_scale, im2col_step=256, remove_center=0):
    n, height, width, channels = value.shape
    if channels != group * group_channels:
        raise ValueError('DCNv4 channel/group mismatch')
    oh = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    ow = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if offset_mask.shape[:3] != (n, oh, ow):
        raise ValueError('DCNv4 offset spatial shape mismatch')
    k = kernel_h * kernel_w - int(remove_center)
    with torch.autocast(value.device.type, enabled=False):
        # CUDA uses float opmath for both FP16/BF16; preserve double for gradcheck.
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        x = value.to(dtype).reshape(n, height * width, group, group_channels).permute(0, 2, 1, 3)
        om = offset_mask[..., :group * k * 3].to(dtype).reshape(n, oh, ow, group, 3 * k)
        offsets = om[..., :2 * k].reshape(n, oh, ow, group, k, 2)
        masks = om[..., 2 * k:]
        yy = torch.arange(oh, dtype=dtype, device=x.device).view(1, oh, 1, 1) * stride_h
        xx = torch.arange(ow, dtype=dtype, device=x.device).view(1, 1, ow, 1) * stride_w
        cy, cx = dilation_h * (kernel_h - 1) // 2, dilation_w * (kernel_w - 1) // 2
        yy = yy + cy - pad_h - cy * offset_scale
        xx = xx + cx - pad_w - cx * offset_scale
        result = torch.zeros((n, oh, ow, group, group_channels), device=x.device, dtype=dtype)
        point = 0
        # The CUDA kernel enumerates width first, then height.
        for i in range(kernel_w):
            for j in range(kernel_h):
                if remove_center and i == kernel_w // 2 and j == kernel_h // 2:
                    continue
                px = xx + (i * dilation_w + offsets[..., point, 0]) * offset_scale
                py = yy + (j * dilation_h + offsets[..., point, 1]) * offset_scale
                x0, y0 = px.floor(), py.floor()
                dx, dy = px - x0, py - y0
                sampled = torch.zeros_like(result)
                for xi, yi, weight in ((x0, y0, (1-dx)*(1-dy)),
                        (x0+1, y0, dx*(1-dy)), (x0, y0+1, (1-dx)*dy),
                        (x0+1, y0+1, dx*dy)):
                    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
                    index = (yi.clamp(0, height-1).long() * width + xi.clamp(0, width-1).long())
                    index = index.permute(0, 3, 1, 2).reshape(n, group, oh*ow, 1)
                    values = x.gather(2, index.expand(-1, -1, -1, group_channels))
                    values = values.reshape(n, group, oh, ow, group_channels).permute(0, 2, 3, 1, 4)
                    sampled = sampled + values * (weight * valid)[..., None]
                result = result + sampled * masks[..., point, None]
                point += 1
    return result.reshape(n, oh, ow, channels).to(value.dtype)


class DCNv4Function:
    apply = staticmethod(dcnv4_reference)
