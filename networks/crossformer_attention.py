import math

import torch
import torch.nn as nn
import torch.nn.functional as F


NEG_INF = -1000000.0


class DynamicPosBias(nn.Module):
    """Predict relative position bias for a grouped attention window."""

    def __init__(self, dim, num_heads, residual=False):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads),
        )

    def forward(self, biases):
        pos = self.pos_proj(biases)
        if self.residual:
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(pos)))
        return pos


class GroupedAttention(nn.Module):
    """CrossFormer-style grouped MHSA with dynamic position bias."""

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        position_bias=True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _build_relative_bias(self, group_h, group_w, device, dtype):
        position_bias_h = torch.arange(1 - group_h, group_h, device=device)
        position_bias_w = torch.arange(1 - group_w, group_w, device=device)
        biases = torch.stack(
            torch.meshgrid(position_bias_h, position_bias_w, indexing="ij")
        )
        biases = biases.flatten(1).transpose(0, 1).contiguous().float()

        coords_h = torch.arange(group_h, device=device)
        coords_w = torch.arange(group_w, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += group_h - 1
        relative_coords[:, :, 1] += group_w - 1
        relative_coords[:, :, 0] *= 2 * group_w - 1
        relative_position_index = relative_coords.sum(-1)

        pos = self.pos(biases)
        relative_position_bias = pos[relative_position_index.view(-1)].view(
            group_h * group_w,
            group_h * group_w,
            -1,
        )
        return relative_position_bias.permute(2, 0, 1).contiguous().to(dtype=dtype)

    def forward(self, x, group_h, group_w, mask=None):
        b_group, num_tokens, channels = x.shape
        assert group_h * group_w == num_tokens, "group shape and token length mismatch"

        qkv = (
            self.qkv(x)
            .reshape(
                b_group,
                num_tokens,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()

        if self.position_bias:
            relative_position_bias = self._build_relative_bias(
                group_h, group_w, attn.device, attn.dtype
            )
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_groups = mask.shape[0]
            attn = (
                attn.view(b_group // num_groups, num_groups, self.num_heads, num_tokens, num_tokens)
                + mask.unsqueeze(1).unsqueeze(0)
            )
            attn = attn.view(-1, self.num_heads, num_tokens, num_tokens)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(b_group, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossFormerLSDAttention(nn.Module):
    """Reusable CrossFormer LSDA attention with dynamic H/W support."""

    def __init__(
        self,
        dim,
        num_heads,
        group_size=7,
        interval=8,
        lsda_flag=0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        position_bias=True,
        adaptive_interval=False,
        pad_type=0,
        no_mask=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.interval = interval
        self.lsda_flag = lsda_flag
        self.adaptive_interval = adaptive_interval
        self.pad_type = pad_type
        self.no_mask = no_mask
        self.attn = GroupedAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            position_bias=position_bias,
        )

    def _infer_hw(self, num_tokens, height, width):
        if height is not None and width is not None:
            return height, width

        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(
                "CrossFormerLSDAttention requires explicit H/W for non-square token maps."
            )
        return side, side

    def _resolve_partition(self, height, width):
        group_size = min(self.group_size, height, width)
        lsda_flag = self.lsda_flag
        if min(height, width) <= group_size:
            lsda_flag = 0

        interval = self.interval
        if self.adaptive_interval:
            interval = max(1, int(math.ceil(height / group_size)))

        return group_size, interval, lsda_flag

    def _build_mask(self, height, width, group_size, interval, lsda_flag, pad_l, pad_r, pad_t, pad_b, device, dtype):
        if self.no_mask:
            return None

        size_div = interval * group_size if lsda_flag == 1 else group_size
        pad_w = (size_div - width % size_div) % size_div
        pad_h = (size_div - height % size_div) % size_div
        if pad_w == 0 and pad_h == 0:
            return None

        padded_h = height + pad_h
        padded_w = width + pad_w
        mask = torch.zeros((1, padded_h, padded_w, 1), device=device, dtype=dtype)
        if pad_h > 0:
            mask[:, -pad_b:, :, :] = -1
            mask[:, :pad_t, :, :] = -1
        if pad_w > 0:
            mask[:, :, -pad_r:, :] = -1
            mask[:, :, :pad_l, :] = -1

        if lsda_flag == 0:
            groups_h = padded_h // group_size
            groups_w = padded_w // group_size
            num_groups = groups_h * groups_w
            mask = (
                mask.reshape(1, groups_h, group_size, groups_w, group_size, 1)
                .permute(0, 1, 3, 2, 4, 5)
                .contiguous()
                .reshape(num_groups, 1, group_size * group_size)
            )
            attn_mask = torch.zeros(
                (num_groups, group_size * group_size, group_size * group_size),
                device=device,
                dtype=dtype,
            )
            return attn_mask.masked_fill(mask < 0, NEG_INF)

        region_h = padded_h // (group_size * interval)
        region_w = padded_w // (group_size * interval)
        num_groups = interval * interval * region_h * region_w
        mask = (
            mask.reshape(1, region_h, group_size, interval, region_w, group_size, interval, 1)
            .permute(0, 1, 4, 3, 6, 2, 5, 7)
            .contiguous()
            .reshape(num_groups, 1, group_size * group_size)
        )
        attn_mask = torch.zeros(
            (num_groups, group_size * group_size, group_size * group_size),
            device=device,
            dtype=dtype,
        )
        return attn_mask.masked_fill(mask < 0, NEG_INF)

    def forward(self, x, H=None, W=None):
        batch_size, num_tokens, channels = x.shape
        height, width = self._infer_hw(num_tokens, H, W)
        if height * width != num_tokens:
            raise ValueError("Token length does not match the provided spatial shape.")

        group_size, interval, lsda_flag = self._resolve_partition(height, width)
        size_div = interval * group_size if lsda_flag == 1 else group_size
        pad_w = (size_div - width % size_div) % size_div
        pad_h = (size_div - height % size_div) % size_div

        if self.pad_type == 0:
            pad_l = pad_t = 0
        else:
            pad_l = pad_w // 2
            pad_t = pad_h // 2
        pad_r = pad_w - pad_l
        pad_b = pad_h - pad_t

        x = x.view(batch_size, height, width, channels)
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, padded_h, padded_w, _ = x.shape

        attn_mask = self._build_mask(
            height,
            width,
            group_size,
            interval,
            lsda_flag,
            pad_l,
            pad_r,
            pad_t,
            pad_b,
            x.device,
            x.dtype,
        )

        if lsda_flag == 0:
            x = (
                x.reshape(batch_size, padded_h // group_size, group_size, padded_w // group_size, group_size, channels)
                .permute(0, 1, 3, 2, 4, 5)
                .contiguous()
                .reshape(batch_size * padded_h * padded_w // (group_size ** 2), group_size * group_size, channels)
            )
            x = self.attn(x, group_size, group_size, mask=attn_mask)
            x = (
                x.reshape(batch_size, padded_h // group_size, padded_w // group_size, group_size, group_size, channels)
                .permute(0, 1, 3, 2, 4, 5)
                .contiguous()
            )
        else:
            region_h = padded_h // (group_size * interval)
            region_w = padded_w // (group_size * interval)
            x = (
                x.reshape(batch_size, region_h, group_size, interval, region_w, group_size, interval, channels)
                .permute(0, 1, 4, 3, 6, 2, 5, 7)
                .contiguous()
                .reshape(batch_size * region_h * region_w * interval * interval, group_size * group_size, channels)
            )
            x = self.attn(x, group_size, group_size, mask=attn_mask)
            x = (
                x.reshape(batch_size, region_h, region_w, interval, interval, group_size, group_size, channels)
                .permute(0, 1, 5, 3, 2, 6, 4, 7)
                .contiguous()
            )

        x = x.view(batch_size, padded_h, padded_w, channels)
        if pad_w > 0 or pad_h > 0:
            x = x[:, pad_t:height + pad_t, pad_l:width + pad_l, :].contiguous()
        return x.view(batch_size, height * width, channels)
