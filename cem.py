import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhanceConv2d(nn.Module):
    """Edge-enhancement convolution with fixed directional kernels and trainable scaling."""

    def __init__(
        self,
        channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        requires_grad=True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "EnhanceConv2d kernel_size must be odd."
        assert channels % 8 == 0, "EnhanceConv2d channels must be a multiple of 8."
        assert channels % groups == 0, "EnhanceConv2d channels must be a multiple of groups."

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if bias and requires_grad:
            self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float32), requires_grad=True)
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "base_weight",
            torch.zeros(channels, channels // groups, kernel_size, kernel_size, dtype=torch.float32),
        )
        self._init_kernels()

        self.sobel_factor = nn.Parameter(
            torch.ones(channels, 1, 1, 1, dtype=torch.float32), requires_grad=requires_grad
        )

    def _init_kernels(self):
        kernel_mid = self.kernel_size // 2
        for idx in range(self.channels):
            weight = self.base_weight[idx]

            if idx % 8 == 0:
                weight[:, 0, :] = -1
                weight[:, 0, kernel_mid] = -2
                weight[:, -1, :] = 1
                weight[:, -1, kernel_mid] = 2
            elif idx % 8 == 1:
                weight[:, :, 0] = -1
                weight[:, kernel_mid, 0] = -2
                weight[:, :, -1] = 1
                weight[:, kernel_mid, -1] = 2
            elif idx % 8 == 2:
                weight[:, 0, 0] = -2
                for i in range(kernel_mid + 1):
                    weight[:, kernel_mid - i, i] = -1
                    weight[:, self.kernel_size - 1 - i, kernel_mid + i] = 1
                weight[:, -1, -1] = 2
            elif idx % 8 == 3:
                weight[:, 0, -1] = -2
                for i in range(kernel_mid + 1):
                    weight[:, i, kernel_mid + i] = -1
                    weight[:, kernel_mid + i, i] = 1
                weight[:, -1, 0] = 2
            elif idx % 8 == 4:
                weight[:, 0, kernel_mid] = 1
                weight[:, kernel_mid, :] = 1
                weight[:, kernel_mid, kernel_mid] = -4
                weight[:, -1, kernel_mid] = 1
            elif idx % 8 == 5:
                weight[:, 0, kernel_mid] = 1
                weight[:, kernel_mid, :] = 1
                weight[:, kernel_mid, kernel_mid] = 4
                weight[:, -1, kernel_mid] = 1
            elif idx % 8 == 6:
                weight[:, 0, :] = -1
                weight[:, -1, :] = 1
            else:
                weight[:, :, 0] = -1
                weight[:, :, -1] = 1

    def forward(self, x):
        weight = self.base_weight * self.sobel_factor
        return F.conv2d(
            x,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class CEM(nn.Module):
    """Contour Enhancement Module.

    Input:
        x: [B, C, H, W]

    Output:
        y: [B, C, H, W]
    """

    def __init__(self, channels, expansion=8, negative_slope=0.1):
        super().__init__()
        assert channels > 0, "channels must be positive."
        assert expansion > 0, "expansion must be positive."

        hidden_channels = channels * expansion
        if hidden_channels % 8 != 0:
            raise ValueError("channels * expansion must be a multiple of 8.")

        self.expand = nn.Conv2d(channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_channels)
        self.expand_act = nn.LeakyReLU(negative_slope, inplace=True)

        self.enhance = EnhanceConv2d(hidden_channels)

        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.project_bn = nn.BatchNorm2d(channels)
        self.project_act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.expand_bn(x)
        x = self.expand_act(x)

        enhanced = self.enhance(x)
        x = x + enhanced

        x = self.project(x)
        x = self.project_bn(x)
        x = self.project_act(x)
        return x + residual


AdaptiveModule3 = CEM


__all__ = ["EnhanceConv2d", "CEM", "AdaptiveModule3"]


if __name__ == "__main__":
    module = CEM(channels=3)
    x = torch.randn(2, 3, 256, 256)
    y = module(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
