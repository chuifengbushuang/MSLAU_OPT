import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_like(x, ref):
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def _normalize_map(x, eps=1e-6):
    x = x.abs()
    scale = x.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return x / scale


def rgb_to_grayscale(x):
    if x.shape[1] == 1:
        return x
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x[:, :3] * weights).sum(dim=1, keepdim=True)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        attention = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * torch.sigmoid(attention)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_map, max_map], dim=1))
        return x * torch.sigmoid(attention)


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class LaplacianEdgePyramid(nn.Module):
    """Builds a MEGANet-style Laplacian edge prior from the raw image."""

    def __init__(self):
        super().__init__()
        gaussian_kernel = torch.tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        ) / 256.0
        laplacian_kernel = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("gaussian_kernel", gaussian_kernel.view(1, 1, 5, 5))
        self.register_buffer("laplacian_kernel", laplacian_kernel.view(1, 1, 3, 3))

    def _apply_kernel(self, x, kernel, padding):
        kernel = kernel.to(dtype=x.dtype, device=x.device).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

    def gaussian_blur(self, x):
        return self._apply_kernel(x, self.gaussian_kernel, padding=2)

    def laplacian(self, x):
        return self._apply_kernel(x, self.laplacian_kernel, padding=1)

    def make_base_edge(self, image):
        gray = rgb_to_grayscale(image)
        level1 = F.avg_pool2d(self.gaussian_blur(gray), kernel_size=2, stride=2)
        level2 = F.avg_pool2d(self.gaussian_blur(level1), kernel_size=2, stride=2)
        laplacian_level1 = level1 - F.interpolate(level2, size=level1.shape[-2:], mode="bilinear", align_corners=False)
        return _normalize_map(laplacian_level1)

    def forward(self, image, target_sizes=None, num_levels=None):
        base_edge = self.make_base_edge(image)
        if target_sizes is not None:
            return [
                F.interpolate(base_edge, size=target_size, mode="bilinear", align_corners=False)
                if base_edge.shape[-2:] != target_size else base_edge
                for target_size in target_sizes
            ]

        if num_levels is None:
            return base_edge

        edges = []
        current = base_edge
        for _ in range(num_levels):
            edges.append(current)
            if min(current.shape[-2:]) <= 1:
                break
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
        return edges

    def for_features(self, image, features):
        return self(image, target_sizes=[feature.shape[-2:] for feature in features])


class EdgeGuidedAttention(nn.Module):
    """
    Adapted from the EGA block in MEGANet.

    Inputs:
    - feature: current feature map [B, C, H, W]
    - coarse_pred: higher-level coarse prediction [B, 1, h, w]
    - image: raw image used to derive the Laplacian edge prior
    - edge_map: optional precomputed 1-channel edge prior
    """

    def __init__(self, channels, reduction=16, use_cbam=True, prediction_is_logit=True):
        super().__init__()
        self.prediction_is_logit = prediction_is_logit
        self.edge_extractor = LaplacianEdgePyramid()
        self.feature_proj = ConvBNReLU(channels, channels, kernel_size=3)
        self.fusion = ConvBNReLU(channels * 3, channels, kernel_size=3)
        self.attention_mask = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True)
        self.cbam = CBAMBlock(channels, reduction=reduction) if use_cbam else nn.Identity()

    def _prepare_prediction(self, coarse_pred, feature):
        coarse_pred = _resize_like(coarse_pred, feature)
        if self.prediction_is_logit:
            coarse_pred = torch.sigmoid(coarse_pred)
        return coarse_pred

    def _prepare_edge(self, feature, image=None, edge_map=None):
        if edge_map is None:
            if image is None:
                raise ValueError("Either image or edge_map must be provided to EdgeGuidedAttention.")
            edge_map = self.edge_extractor(image, target_sizes=[feature.shape[-2:]])[0]
        else:
            edge_map = _resize_like(edge_map, feature)
        return _normalize_map(edge_map)

    def forward(self, feature, coarse_pred, image=None, edge_map=None, return_aux=False):
        feature = self.feature_proj(feature)
        coarse_pred = self._prepare_prediction(coarse_pred, feature)
        edge_map = self._prepare_edge(feature, image=image, edge_map=edge_map)

        boundary_attention = _normalize_map(self.edge_extractor.laplacian(coarse_pred))
        reverse_attention = 1.0 - coarse_pred

        fused = self.fusion(
            torch.cat(
                [
                    edge_map * feature,
                    boundary_attention * feature,
                    reverse_attention * feature,
                ],
                dim=1,
            )
        )

        attention = torch.sigmoid(self.attention_mask(fused))
        refined = feature + fused * attention
        refined = self.cbam(refined)

        if not return_aux:
            return refined

        return refined, {
            "edge_map": edge_map,
            "boundary_attention": boundary_attention,
            "reverse_attention": reverse_attention,
            "attention_mask": attention,
        }


__all__ = [
    "CBAMBlock",
    "ConvBNReLU",
    "EdgeGuidedAttention",
    "LaplacianEdgePyramid",
    "rgb_to_grayscale",
]
