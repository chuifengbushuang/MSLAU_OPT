# All rights reserved.
from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath, to_2tuple
import copy

from networks.msla import MSLA
from networks.edge_guidance import EdgeGuidedAttention
from networks.naf_select_reg import PyramidNAF2D
from networks.naf_discrete import CascadeDiscreteNAF2D

layer_scale = False
init_value = 1e-6


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LFE(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GFE(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        # in_channels, key_channels, head_count, value_channels
        self.attn = MSLA(
            dim=dim, num_heads=num_heads
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.ls:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    
class Encoder(nn.Module):

    def __init__(self, depth=[4, 8, 11, 5], img_size=224, in_chans=3, num_classes=1, embed_dim=[64, 128, 256, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            LFE(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            LFE(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            GFE(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            GFE(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
        for i in range(depth[3])])
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        features = []
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        features.append(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        features.append(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        features.append(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        features.append(x)
        return features


class SpatialFeatureAdapter(nn.Module):
    """Legacy P5 residual adapter applied after a complete Hiera stage."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(32, channels // reduction)
        self.norm = nn.GroupNorm(1, channels)
        self.down = nn.Conv2d(channels, hidden_channels, 1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden_channels, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


class HieraBlockAdapter(nn.Module):
    """Official SAM2-UNet prompt adapter applied before each Hiera block."""

    def __init__(self, block, bottleneck=32):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x + self.prompt_learn(x))


class HieraBlockAdapterMAFA(nn.Module):
    """MAFA-inspired prompt adapter applied before each Hiera block.

    Multi-scale convolution branches replace MAFA's frequency filtering, and an
    SKNet-style channel attention fuses the branch responses adaptively instead
    of a plain concatenation. The residual output projection is zero
    initialized so the adapter starts as an identity map over the frozen Hiera
    features.
    """

    STAGE_KERNELS = {
        144: (1, 3, 5, 7),
        288: (1, 3, 5, 7),
        576: (1, 3),
        1152: (1, 3),
    }

    def __init__(self, block, bottleneck=32, branch_kernels=(1, 3, 5, 7),
                 reduction=2, stage_adaptive=False):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        if stage_adaptive:
            branch_kernels = self.STAGE_KERNELS.get(dim, branch_kernels)
        self.bottleneck = bottleneck
        self.branch_kernels = tuple(branch_kernels)
        self.num_branches = len(self.branch_kernels)
        self.down = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1),
            nn.GELU(),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bottleneck, bottleneck, kernel, padding=kernel // 2),
                nn.GELU(),
            )
            for kernel in self.branch_kernels
        ])
        hidden = max(8, bottleneck // reduction)
        self.fusion_fc1 = nn.Conv2d(bottleneck, hidden, 1)
        self.fusion_fc2 = nn.Conv2d(hidden, self.num_branches * bottleneck, 1)
        self.up = nn.Conv2d(bottleneck, dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # Hiera operates on channels-last tensors: (B, H, W, C).
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        feat = self.down(x_nchw)
        branch_outputs = [branch(feat) for branch in self.branches]

        descriptor = branch_outputs[0]
        for branch_output in branch_outputs[1:]:
            descriptor = descriptor + branch_output
        descriptor = F.adaptive_avg_pool2d(descriptor, 1)
        descriptor = self.fusion_fc2(F.gelu(self.fusion_fc1(descriptor)))
        weights = descriptor.view(
            -1, self.num_branches, self.bottleneck, 1, 1)
        weights = torch.softmax(weights, dim=1)

        fused = torch.zeros_like(branch_outputs[0])
        for index in range(self.num_branches):
            fused = fused + weights[:, index] * branch_outputs[index]

        delta = self.up(fused).permute(0, 2, 3, 1).contiguous()
        return self.block(x + delta)


class HieraBlockAdapterMAFAv2(nn.Module):
    """MAFA-inspired prompt adapter (fusion v2).

    Same stage-adaptive multi-scale conv branches as the v1 MAFA adapter, but
    the fusion step is upgraded following Plan A:
      - per-branch dual-pooling descriptor [GAP ; GMP] (keeps branch identity
        and adds a peak-response cue that favours small salient objects),
      - descriptor concatenated over branches,
      - LayerNorm + a wider selection MLP (hidden=64),
      - softmax selection across branches, zero-init residual output.
    No frequency filtering (removed vs. the original MAFA).
    """

    STAGE_KERNELS = {
        144: (1, 3, 5, 7),
        288: (1, 3, 5, 7),
        576: (1, 3),
        1152: (1, 3),
    }

    def __init__(self, block, bottleneck=32, branch_kernels=(1, 3, 5, 7),
                 stage_adaptive=True, hidden=64):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        if stage_adaptive:
            branch_kernels = self.STAGE_KERNELS.get(dim, branch_kernels)
        self.bottleneck = bottleneck
        self.branch_kernels = tuple(branch_kernels)
        self.num_branches = len(self.branch_kernels)

        self.down = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1),
            nn.GELU(),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bottleneck, bottleneck, kernel, padding=kernel // 2),
                nn.GELU(),
            )
            for kernel in self.branch_kernels
        ])

        descriptor_dim = 2 * self.num_branches * bottleneck
        self.fusion_norm = nn.LayerNorm(descriptor_dim)
        self.fusion_fc1 = nn.Linear(descriptor_dim, hidden)
        self.fusion_fc2 = nn.Linear(hidden, self.num_branches * bottleneck)

        self.up = nn.Conv2d(bottleneck, dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # Hiera operates on channels-last tensors: (B, H, W, C).
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        feat = self.down(x_nchw)
        branch_outputs = [branch(feat) for branch in self.branches]

        descriptors = []
        for branch_output in branch_outputs:
            gap = F.adaptive_avg_pool2d(branch_output, 1).flatten(1)
            gmp = branch_output.amax(dim=(2, 3))
            descriptors.append(torch.cat([gap, gmp], dim=1))
        descriptor = torch.cat(descriptors, dim=1)

        descriptor = self.fusion_norm(descriptor)
        descriptor = self.fusion_fc2(F.gelu(self.fusion_fc1(descriptor)))
        weights = descriptor.view(
            -1, self.num_branches, self.bottleneck, 1, 1)
        weights = torch.softmax(weights, dim=1)

        fused = torch.zeros_like(branch_outputs[0])
        for index in range(self.num_branches):
            fused = fused + weights[:, index] * branch_outputs[index]

        delta = self.up(fused).permute(0, 2, 3, 1).contiguous()
        return self.block(x + delta)


class HieraBlockAdapterMAFAv3(nn.Module):
    """MAFA-inspired prompt adapter (fusion v3).

    Fusion module = v2 channel-selective fusion (per-branch [GAP;GMP]
    descriptor + LayerNorm + hidden=64 MLP) augmented with a lightweight
    per-pixel spatial branch gate, so large structures can locally re-weight
    the multi-scale branches instead of relying on a single global choice.
    No frequency filtering. Zero-init residual output.
    """

    STAGE_KERNELS = {
        144: (1, 3, 5, 7),
        288: (1, 3, 5, 7),
        576: (1, 3),
        1152: (1, 3),
    }

    def __init__(self, block, bottleneck=32, branch_kernels=(1, 3, 5, 7),
                 stage_adaptive=True, hidden=64):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        if stage_adaptive:
            branch_kernels = self.STAGE_KERNELS.get(dim, branch_kernels)
        self.bottleneck = bottleneck
        self.branch_kernels = tuple(branch_kernels)
        self.num_branches = len(self.branch_kernels)

        self.down = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1),
            nn.GELU(),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bottleneck, bottleneck, kernel, padding=kernel // 2),
                nn.GELU(),
            )
            for kernel in self.branch_kernels
        ])

        descriptor_dim = 2 * self.num_branches * bottleneck
        self.fusion_norm = nn.LayerNorm(descriptor_dim)
        self.fusion_fc1 = nn.Linear(descriptor_dim, hidden)
        self.fusion_fc2 = nn.Linear(hidden, self.num_branches * bottleneck)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.num_branches * bottleneck, self.num_branches, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.spatial_gate[0].weight)
        nn.init.zeros_(self.spatial_gate[0].bias)

        self.up = nn.Conv2d(bottleneck, dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # Hiera operates on channels-last tensors: (B, H, W, C).
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        feat = self.down(x_nchw)
        branch_outputs = [branch(feat) for branch in self.branches]

        descriptors = []
        for branch_output in branch_outputs:
            gap = F.adaptive_avg_pool2d(branch_output, 1).flatten(1)
            gmp = branch_output.amax(dim=(2, 3))
            descriptors.append(torch.cat([gap, gmp], dim=1))
        descriptor = torch.cat(descriptors, dim=1)

        descriptor = self.fusion_norm(descriptor)
        descriptor = self.fusion_fc2(F.gelu(self.fusion_fc1(descriptor)))
        a_chan = torch.softmax(
            descriptor.view(-1, self.num_branches, self.bottleneck, 1, 1),
            dim=1)

        concat = torch.cat(branch_outputs, dim=1)
        g_spat = self.spatial_gate(concat).unsqueeze(2)
        weights = a_chan * g_spat

        fused = torch.zeros_like(branch_outputs[0])
        for index in range(self.num_branches):
            fused = fused + weights[:, index] * branch_outputs[index]

        delta = self.up(fused).permute(0, 2, 3, 1).contiguous()
        return self.block(x + delta)


class HieraBlockAdapterMAFAv4(nn.Module):
    """MAFA-inspired prompt adapter (fusion B / v4).

    Fusion v2 (per-branch [GAP;GMP] descriptor + LayerNorm + hidden=64 MLP,
    stage-adaptive multi-scale branches, zero-init residual) with one change:
    the GMP (max-pooling) descriptor of each branch is multiplied by a learnable
    per-branch scalar initialised to 1. This lets the model decide how much to
    rely on the peak-response cue, which otherwise biases branch selection
    towards small objects. Capacity overhead: num_branches params per block.
    """

    STAGE_KERNELS = {
        144: (1, 3, 5, 7),
        288: (1, 3, 5, 7),
        576: (1, 3),
        1152: (1, 3),
    }

    def __init__(self, block, bottleneck=32, branch_kernels=(1, 3, 5, 7),
                 stage_adaptive=True, hidden=64):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        if stage_adaptive:
            branch_kernels = self.STAGE_KERNELS.get(dim, branch_kernels)
        self.bottleneck = bottleneck
        self.branch_kernels = tuple(branch_kernels)
        self.num_branches = len(self.branch_kernels)

        self.down = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1),
            nn.GELU(),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bottleneck, bottleneck, kernel, padding=kernel // 2),
                nn.GELU(),
            )
            for kernel in self.branch_kernels
        ])

        descriptor_dim = 2 * self.num_branches * bottleneck
        self.fusion_norm = nn.LayerNorm(descriptor_dim)
        self.fusion_fc1 = nn.Linear(descriptor_dim, hidden)
        self.fusion_fc2 = nn.Linear(hidden, self.num_branches * bottleneck)
        self.gmp_scale = nn.Parameter(torch.ones(self.num_branches))

        self.up = nn.Conv2d(bottleneck, dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # Hiera operates on channels-last tensors: (B, H, W, C).
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        feat = self.down(x_nchw)
        branch_outputs = [branch(feat) for branch in self.branches]

        descriptors = []
        for index, branch_output in enumerate(branch_outputs):
            gap = F.adaptive_avg_pool2d(branch_output, 1).flatten(1)
            gmp = branch_output.amax(dim=(2, 3))
            descriptors.append(torch.cat(
                [gap, self.gmp_scale[index] * gmp], dim=1))
        descriptor = torch.cat(descriptors, dim=1)

        descriptor = self.fusion_norm(descriptor)
        descriptor = self.fusion_fc2(F.gelu(self.fusion_fc1(descriptor)))
        weights = torch.softmax(
            descriptor.view(-1, self.num_branches, self.bottleneck, 1, 1),
            dim=1)

        fused = torch.zeros_like(branch_outputs[0])
        for index in range(self.num_branches):
            fused = fused + weights[:, index] * branch_outputs[index]

        delta = self.up(fused).permute(0, 2, 3, 1).contiguous()
        return self.block(x + delta)


class HieraBlockAdapterMAFAv5(nn.Module):
    """MAFA-inspired prompt adapter (fusion v5).

    True multi-scale processing followed by adaptive fusion:
      - branches: 1x1 local, 3x3 depthwise, 3x3 dilated (rate=6) depthwise,
        and a global-average-pooled branch (image-level context);
      - channel SKNet over per-branch [GAP;GMP] descriptors (as in v2);
      - parameter-free spatial adaptivity: per-pixel/per-channel softmax
        temperature driven by the cross-branch variance (one learnable scalar,
        init 0 -> starts identical to the global v2 selection);
      - identity-preserving residual (+ id_scale * feat, init 0) to counter the
        area-shrinking / recall-drop bias of convex fusion.
    Residual output projection is zero-initialised.
    """

    def __init__(self, block, bottleneck=32, hidden=64, dilation=6):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        self.bottleneck = bottleneck
        self.num_branches = 4

        self.down = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1),
            nn.GELU(),
        )
        self.branch_local = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, 1),
            nn.GELU(),
        )
        self.branch_dw = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1,
                      groups=bottleneck, bias=False),
            nn.Conv2d(bottleneck, bottleneck, 1),
            nn.GELU(),
        )
        self.branch_dil = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, 3, padding=dilation,
                      dilation=dilation, groups=bottleneck, bias=False),
            nn.Conv2d(bottleneck, bottleneck, 1),
            nn.GELU(),
        )
        self.branch_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(bottleneck, bottleneck, 1),
            nn.GELU(),
        )

        descriptor_dim = 2 * self.num_branches * bottleneck
        self.fusion_norm = nn.LayerNorm(descriptor_dim)
        self.fusion_fc1 = nn.Linear(descriptor_dim, hidden)
        self.fusion_fc2 = nn.Linear(hidden, self.num_branches * bottleneck)

        self.var_scale = nn.Parameter(torch.zeros(1))
        self.id_scale = nn.Parameter(torch.zeros(1))

        self.up = nn.Conv2d(bottleneck, dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # Hiera operates on channels-last tensors: (B, H, W, C).
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        feat = self.down(x_nchw)
        branch_outputs = [
            self.branch_local(feat),
            self.branch_dw(feat),
            self.branch_dil(feat),
            self.branch_global(feat).expand_as(feat),
        ]

        descriptors = []
        for branch_output in branch_outputs:
            gap = F.adaptive_avg_pool2d(branch_output, 1).flatten(1)
            gmp = branch_output.amax(dim=(2, 3))
            descriptors.append(torch.cat([gap, gmp], dim=1))
        descriptor = torch.cat(descriptors, dim=1)
        descriptor = self.fusion_norm(descriptor)
        descriptor = self.fusion_fc2(F.gelu(self.fusion_fc1(descriptor)))
        logits = descriptor.view(
            -1, self.num_branches, self.bottleneck, 1, 1)

        stacked = torch.stack(branch_outputs, dim=1)
        variance = stacked.var(dim=1, unbiased=False)
        normalised = variance / (
            variance.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
        temperature = 1.0 + self.var_scale * normalised.unsqueeze(1)
        weights = torch.softmax(logits / temperature, dim=1)

        fused = torch.zeros_like(branch_outputs[0])
        for index in range(self.num_branches):
            fused = fused + weights[:, index] * branch_outputs[index]
        fused = fused + self.id_scale * feat

        delta = self.up(fused).permute(0, 2, 3, 1).contiguous()
        return self.block(x + delta)


class HieraBlockAdapterMAFA_MR(nn.Module):
    """MAFA-inspired prompt adapter (2a: multi-resolution branches).

    Keeps the v2 channel SKNet fusion (per-branch [GAP;GMP] descriptor +
    LayerNorm + hidden=64 MLP + softmax over branches + zero-init residual),
    but replaces the kernel-size branches with genuine multi-resolution
    branches: 1x1 | 3x3@1x | 3x3@1/2 | 3x3@1/4. Branch outputs are brought
    back to the input resolution (bilinear) before fusion.
    """

    def __init__(self, block, bottleneck=32, hidden=64):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        self.bottleneck = bottleneck
        self.num_branches = 4

        self.down = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1),
            nn.GELU(),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, 1),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, bias=False),
            nn.GELU(),
        )

        descriptor_dim = 2 * self.num_branches * bottleneck
        self.fusion_norm = nn.LayerNorm(descriptor_dim)
        self.fusion_fc1 = nn.Linear(descriptor_dim, hidden)
        self.fusion_fc2 = nn.Linear(hidden, self.num_branches * bottleneck)

        self.up = nn.Conv2d(bottleneck, dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # Hiera operates on channels-last tensors: (B, H, W, C).
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        feat = self.down(x_nchw)
        size = feat.shape[-2:]
        branch_outputs = [
            self.branch1(feat),
            self.branch2(feat),
            F.interpolate(self.branch3(feat), size=size,
                          mode="bilinear", align_corners=False),
            F.interpolate(self.branch4(feat), size=size,
                          mode="bilinear", align_corners=False),
        ]

        descriptors = []
        for branch_output in branch_outputs:
            gap = F.adaptive_avg_pool2d(branch_output, 1).flatten(1)
            gmp = branch_output.amax(dim=(2, 3))
            descriptors.append(torch.cat([gap, gmp], dim=1))
        descriptor = torch.cat(descriptors, dim=1)
        descriptor = self.fusion_norm(descriptor)
        descriptor = self.fusion_fc2(F.gelu(self.fusion_fc1(descriptor)))
        weights = torch.softmax(
            descriptor.view(-1, self.num_branches, self.bottleneck, 1, 1),
            dim=1)

        fused = torch.zeros_like(branch_outputs[0])
        for index in range(self.num_branches):
            fused = fused + weights[:, index] * branch_outputs[index]

        delta = self.up(fused).permute(0, 2, 3, 1).contiguous()
        return self.block(x + delta)


class HieraBlockAdapterMAFAv2GN(nn.Module):
    """MAFA fusion v2 + 3a branch normalization.

    Same as the v2 adapter (stage-adaptive multi-scale branches, per-branch
    [GAP;GMP] SKNet descriptor, LayerNorm + hidden=64 MLP, softmax over
    branches, zero-init residual), but each branch output is normalised with a
    per-sample GroupNorm *before* the weighted sum. The selection descriptor
    still uses the un-normalised branch statistics. This equalises branch
    magnitudes so the convex weighted sum does not shrink the feature
    (anti area-collapse). No BatchNorm, no frequency filtering.
    """

    STAGE_KERNELS = {
        144: (1, 3, 5, 7),
        288: (1, 3, 5, 7),
        576: (1, 3),
        1152: (1, 3),
    }

    def __init__(self, block, bottleneck=32, branch_kernels=(1, 3, 5, 7),
                 stage_adaptive=True, hidden=64, num_groups=8):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        if stage_adaptive:
            branch_kernels = self.STAGE_KERNELS.get(dim, branch_kernels)
        self.bottleneck = bottleneck
        self.branch_kernels = tuple(branch_kernels)
        self.num_branches = len(self.branch_kernels)

        self.down = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1),
            nn.GELU(),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bottleneck, bottleneck, kernel, padding=kernel // 2),
                nn.GELU(),
            )
            for kernel in self.branch_kernels
        ])
        self.branch_norms = nn.ModuleList([
            nn.GroupNorm(num_groups, bottleneck)
            for _ in range(self.num_branches)
        ])

        descriptor_dim = 2 * self.num_branches * bottleneck
        self.fusion_norm = nn.LayerNorm(descriptor_dim)
        self.fusion_fc1 = nn.Linear(descriptor_dim, hidden)
        self.fusion_fc2 = nn.Linear(hidden, self.num_branches * bottleneck)

        self.up = nn.Conv2d(bottleneck, dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # Hiera operates on channels-last tensors: (B, H, W, C).
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        feat = self.down(x_nchw)
        branch_outputs = [branch(feat) for branch in self.branches]

        descriptors = []
        for branch_output in branch_outputs:
            gap = F.adaptive_avg_pool2d(branch_output, 1).flatten(1)
            gmp = branch_output.amax(dim=(2, 3))
            descriptors.append(torch.cat([gap, gmp], dim=1))
        descriptor = torch.cat(descriptors, dim=1)
        descriptor = self.fusion_norm(descriptor)
        descriptor = self.fusion_fc2(F.gelu(self.fusion_fc1(descriptor)))
        weights = torch.softmax(
            descriptor.view(-1, self.num_branches, self.bottleneck, 1, 1),
            dim=1)

        fused = torch.zeros_like(branch_outputs[0])
        for index in range(self.num_branches):
            normalised = self.branch_norms[index](branch_outputs[index])
            fused = fused + weights[:, index] * normalised

        delta = self.up(fused).permute(0, 2, 3, 1).contiguous()
        return self.block(x + delta)


class HieraBlockAdapterMAFAv2AR(nn.Module):
    """MAFA fusion v2 + A: anchor-residual with independent sigmoid gates.

    Same stage-adaptive branches and [GAP;GMP] SKNet descriptor as v2, but the
    combination rule changes from a convex softmax mixture
        fused = sum_i a_i F_i   (sum_i a_i = 1)
    to an anchor-residual form with independent (non-normalised) sigmoid gates
        fused = feat + sum_i g_i * (F_i - feat),   g_i = sigmoid(.)
    which preserves the projected feature magnitude and can amplify or shrink
    branch contributions, countering the area-collapse / large-object bias of
    convex fusion. Gate bias is initialised negative so the module starts close
    to the identity anchor. Residual output is zero-initialised.
    """

    STAGE_KERNELS = {
        144: (1, 3, 5, 7),
        288: (1, 3, 5, 7),
        576: (1, 3),
        1152: (1, 3),
    }

    def __init__(self, block, bottleneck=32, branch_kernels=(1, 3, 5, 7),
                 stage_adaptive=True, hidden=64, gate_bias=-2.0):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        if stage_adaptive:
            branch_kernels = self.STAGE_KERNELS.get(dim, branch_kernels)
        self.bottleneck = bottleneck
        self.branch_kernels = tuple(branch_kernels)
        self.num_branches = len(self.branch_kernels)

        self.down = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1),
            nn.GELU(),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bottleneck, bottleneck, kernel, padding=kernel // 2),
                nn.GELU(),
            )
            for kernel in self.branch_kernels
        ])

        descriptor_dim = 2 * self.num_branches * bottleneck
        self.fusion_norm = nn.LayerNorm(descriptor_dim)
        self.fusion_fc1 = nn.Linear(descriptor_dim, hidden)
        self.fusion_fc2 = nn.Linear(hidden, self.num_branches * bottleneck)
        nn.init.constant_(self.fusion_fc2.bias, gate_bias)

        self.up = nn.Conv2d(bottleneck, dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # Hiera operates on channels-last tensors: (B, H, W, C).
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        feat = self.down(x_nchw)
        branch_outputs = [branch(feat) for branch in self.branches]

        descriptors = []
        for branch_output in branch_outputs:
            gap = F.adaptive_avg_pool2d(branch_output, 1).flatten(1)
            gmp = branch_output.amax(dim=(2, 3))
            descriptors.append(torch.cat([gap, gmp], dim=1))
        descriptor = torch.cat(descriptors, dim=1)
        descriptor = self.fusion_norm(descriptor)
        descriptor = self.fusion_fc2(F.gelu(self.fusion_fc1(descriptor)))
        gates = torch.sigmoid(
            descriptor.view(-1, self.num_branches, self.bottleneck, 1, 1))

        fused = feat
        for index in range(self.num_branches):
            fused = fused + gates[:, index] * (branch_outputs[index] - feat)

        delta = self.up(fused).permute(0, 2, 3, 1).contiguous()
        return self.block(x + delta)


class HieraFeatureEncoder(nn.Module):
    """Frozen SAM2 Hiera backbone with selectable trainable adapters."""

    CHANNELS = {
        "sam2_hiera_large": (144, 288, 576, 1152),
    }

    def __init__(self, model_name="sam2_hiera_large", checkpoint_path=None,
                 freeze_backbone=True, adapter_mode="stage_output",
                 adapter_reduction=4):
        super().__init__()
        import timm

        if model_name not in self.CHANNELS:
            raise ValueError("Unsupported Hiera encoder: {}".format(model_name))
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.adapter_mode = adapter_mode
        self.out_channels = self.CHANNELS[model_name]
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
        )
        if checkpoint_path:
            self._load_backbone_checkpoint(checkpoint_path)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        if adapter_mode == "stage_output":
            self.adapters = nn.ModuleList([
                SpatialFeatureAdapter(channels, reduction=adapter_reduction)
                for channels in self.out_channels
            ])
        elif adapter_mode == "block_prompt":
            self.adapters = None
            self.backbone.model.blocks = nn.ModuleList([
                HieraBlockAdapter(block) for block in self.backbone.model.blocks
            ])
        elif adapter_mode == "block_mafa":
            self.adapters = None
            self.backbone.model.blocks = nn.ModuleList([
                HieraBlockAdapterMAFA(block)
                for block in self.backbone.model.blocks
            ])
        elif adapter_mode == "block_mafa_stage":
            self.adapters = None
            self.backbone.model.blocks = nn.ModuleList([
                HieraBlockAdapterMAFA(block, stage_adaptive=True)
                for block in self.backbone.model.blocks
            ])
        elif adapter_mode == "block_mafa_v2":
            self.adapters = None
            self.backbone.model.blocks = nn.ModuleList([
                HieraBlockAdapterMAFAv2(block, stage_adaptive=True, hidden=64)
                for block in self.backbone.model.blocks
            ])
        elif adapter_mode == "block_mafa_v3":
            self.adapters = None
            self.backbone.model.blocks = nn.ModuleList([
                HieraBlockAdapterMAFAv3(block, stage_adaptive=True, hidden=64)
                for block in self.backbone.model.blocks
            ])
        elif adapter_mode == "block_mafa_v4":
            self.adapters = None
            self.backbone.model.blocks = nn.ModuleList([
                HieraBlockAdapterMAFAv4(block, stage_adaptive=True, hidden=64)
                for block in self.backbone.model.blocks
            ])
        elif adapter_mode == "block_mafa_v5":
            self.adapters = None
            self.backbone.model.blocks = nn.ModuleList([
                HieraBlockAdapterMAFAv5(block, hidden=64)
                for block in self.backbone.model.blocks
            ])
        elif adapter_mode == "block_mafa_mr":
            self.adapters = None
            self.backbone.model.blocks = nn.ModuleList([
                HieraBlockAdapterMAFA_MR(block, hidden=64)
                for block in self.backbone.model.blocks
            ])
        elif adapter_mode == "block_mafa_v2gn":
            self.adapters = None
            self.backbone.model.blocks = nn.ModuleList([
                HieraBlockAdapterMAFAv2GN(block, stage_adaptive=True, hidden=64)
                for block in self.backbone.model.blocks
            ])
        elif adapter_mode == "block_mafa_v2ar":
            self.adapters = None
            self.backbone.model.blocks = nn.ModuleList([
                HieraBlockAdapterMAFAv2AR(block, stage_adaptive=True, hidden=64)
                for block in self.backbone.model.blocks
            ])
        else:
            raise ValueError("Unsupported Hiera adapter mode: {}".format(adapter_mode))

    def _load_backbone_checkpoint(self, checkpoint_path):
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path, device="cpu")
        else:
            state_dict = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = state_dict.get("state_dict", state_dict)
            state_dict = state_dict.get("model", state_dict)

        backbone_state = self.backbone.state_dict()
        mapped_state = {}
        for key, value in state_dict.items():
            mapped_key = key if key.startswith("model.") else "model." + key
            if mapped_key in backbone_state:
                mapped_state[mapped_key] = value
        missing_keys = sorted(set(backbone_state) - set(mapped_state))
        if missing_keys:
            raise RuntimeError(
                "Hiera checkpoint is incomplete; missing {} keys, first: {}".format(
                    len(missing_keys), missing_keys[:5]))
        self.backbone.load_state_dict(mapped_state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        if self.freeze_backbone and self.adapter_mode == "stage_output":
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        if self.adapters is None:
            return features
        return [adapter(feature) for adapter, feature in zip(self.adapters, features)]

class Conv_MLA(nn.Module):
    def __init__(self, embed_dim=[64, 128, 256, 512], mla_channels=64, norm_cfg=None,
                 fusion_mode="fixed"):
        super(Conv_MLA, self).__init__()

        if fusion_mode not in {"fixed", "lff_scale"}:
            raise ValueError("Unsupported fusion mode: {}".format(fusion_mode))
        self.fusion_mode = fusion_mode
        if fusion_mode == "lff_scale":
            # Rows follow encoder stage order: shallow stage 1 -> deep stage 4.
            # Scaled softmax starts at gamma=1 and keeps each channel's sum at 4.
            self.fusion_logits = nn.Parameter(torch.zeros(4, mla_channels))
        else:
            self.register_parameter("fusion_logits", None)

        self.mla_p4 = nn.Sequential(nn.Conv2d(embed_dim[1], mla_channels, 1 ,bias=False),
                                    nn.BatchNorm2d(mla_channels), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                    )
        self.mla_p3 = nn.Sequential(nn.Conv2d(embed_dim[2], embed_dim[1], 1 ,bias=False),
                                    nn.BatchNorm2d(embed_dim[1]), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(embed_dim[1], mla_channels, 1, bias=False),
                                    nn.BatchNorm2d(mla_channels), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                    )
        self.mla_p2 = nn.Sequential(nn.Conv2d(embed_dim[3], embed_dim[2], 1 ,bias=False),
                                    nn.BatchNorm2d(embed_dim[2]), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(embed_dim[2], embed_dim[1], 1, bias=False),
                                    nn.BatchNorm2d(embed_dim[1]), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(embed_dim[1], mla_channels, 1, bias=False),
                                    nn.BatchNorm2d(mla_channels), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                    )

        self.mla_p2_3x3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                    bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p3_3x3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                    bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p4_3x3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                    bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p5_3x3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                    bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())

    def forward(self,features):

        uni5, uni4, uni3, uni2 = features

        uni_mla_p4 = self.mla_p4(uni4)
        uni_mla_p3 = self.mla_p3(uni3)
        uni_mla_p2 = self.mla_p2(uni2)

        if self.fusion_mode == "lff_scale":
            gamma = 4.0 * torch.softmax(self.fusion_logits, dim=0)
            uni5 = uni5 * gamma[0].view(1, -1, 1, 1)
            uni_mla_p4 = uni_mla_p4 * gamma[1].view(1, -1, 1, 1)
            uni_mla_p3 = uni_mla_p3 * gamma[2].view(1, -1, 1, 1)
            uni_mla_p2 = uni_mla_p2 * gamma[3].view(1, -1, 1, 1)

        mla_p4_plus = uni5 + uni_mla_p4
        mla_p3_plus = mla_p4_plus + uni_mla_p3
        mla_p2_plus = mla_p3_plus + uni_mla_p2

        mla_p5 = self.mla_p5_3x3(uni5)
        mla_p4 = self.mla_p4_3x3(mla_p4_plus)
        mla_p3 = self.mla_p3_3x3(mla_p3_plus)
        mla_p2 = self.mla_p2_3x3(mla_p2_plus)

        return [mla_p5, mla_p4, mla_p3, mla_p2]

class MLAHead(nn.Module):
    def __init__(self, mla_channels=64):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.head3 = nn.Sequential(
            nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.head4 = nn.Sequential(
            nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.head5 = nn.Sequential(
            nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels), nn.ReLU())

    def forward(self, mla_list):
        head5 = F.interpolate(self.head2(
            mla_list[0]), 2*mla_list[0].shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head3(
            mla_list[1]), 2*mla_list[1].shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head4(
            mla_list[2]), 2*mla_list[2].shape[-1], mode='bilinear', align_corners=True)
        head2 = F.interpolate(self.head5(
            mla_list[3]), 2*mla_list[3].shape[-1], mode='bilinear', align_corners=True)
        return torch.cat([head5, head4, head3, head2], dim=1)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 dilation=1, groups=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class ResidualRefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.refine(x))


class MultiScaleContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = ConvNormAct(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch1 = ConvNormAct(
            out_channels, out_channels, groups=out_channels)
        self.branch2 = ConvNormAct(
            out_channels, out_channels, padding=2, dilation=2, groups=out_channels)
        self.branch3 = ConvNormAct(
            out_channels, out_channels, padding=3, dilation=3, groups=out_channels)
        self.fuse = ConvNormAct(3 * out_channels, out_channels, kernel_size=1, padding=0)
        self.refine = ResidualRefineBlock(out_channels)

    def forward(self, x):
        x = self.project(x)
        context = self.fuse(torch.cat([
            self.branch1(x), self.branch2(x), self.branch3(x)
        ], dim=1))
        return self.refine(x + context)


class HaarWaveletEdgeHead(nn.Module):
    """Parameter-free two-level Haar high-frequency extractor."""

    def __init__(self):
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ).unsqueeze(1) / 2.0
        self.register_buffer("haar_filters", filters, persistent=False)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, normalized_image):
        if normalized_image.shape[1] == 1:
            image = normalized_image.repeat(1, 3, 1, 1)
        else:
            image = normalized_image[:, :3]
        image = (image * self.image_std + self.image_mean).clamp(0.0, 1.0)
        gray = (
            0.299 * image[:, 0:1]
            + 0.587 * image[:, 1:2]
            + 0.114 * image[:, 2:3]
        )

        level1 = F.conv2d(gray, self.haar_filters, stride=2)
        level2 = F.conv2d(level1[:, :1], self.haar_filters, stride=2)
        high1 = level1[:, 1:].abs()
        high2 = F.interpolate(
            level2[:, 1:].abs(), size=high1.shape[-2:],
            mode="bilinear", align_corners=False)
        edges = torch.cat([high1, high2], dim=1)
        scale = edges.flatten(2).amax(dim=2).view(edges.shape[0], 6, 1, 1)
        return edges / scale.clamp_min(1e-6)


class ProgressiveFusionBlock(nn.Module):
    def __init__(self, skip_channels, channels, edge_channels=6):
        super().__init__()
        self.skip_project = ConvNormAct(
            skip_channels, channels, kernel_size=1, padding=0)
        self.decoder_project = ConvNormAct(
            channels, channels, kernel_size=1, padding=0)
        hidden_channels = max(16, channels // 4)
        self.selective_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )
        self.edge_gate = nn.Sequential(
            nn.Conv2d(edge_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.edge_scale = nn.Parameter(torch.tensor(0.1))
        self.channel_scale = nn.Parameter(torch.tensor(0.1))
        self.reverse_scale = nn.Parameter(torch.tensor(0.1))
        self.refine = ResidualRefineBlock(channels)

    def forward(self, skip, decoder, wavelet_edges=None, reverse_attention=None):
        decoder = F.interpolate(
            decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        decoder = self.decoder_project(decoder)
        skip = self.skip_project(skip)
        gate = self.selective_gate(torch.cat([skip, decoder], dim=1))
        fused = decoder + gate * skip
        fused = fused * (1.0 + self.channel_scale * self.channel_gate(fused))

        if wavelet_edges is not None:
            edges = F.interpolate(
                wavelet_edges, size=fused.shape[-2:],
                mode="bilinear", align_corners=False)
            fused = fused * (1.0 + self.edge_scale * self.edge_gate(edges))

        if reverse_attention is not None:
            reverse_attention = F.interpolate(
                reverse_attention, size=fused.shape[-2:],
                mode="bilinear", align_corners=False)
            fused = fused * (1.0 + self.reverse_scale * reverse_attention)
        return self.refine(fused)


class ProgressiveWaveletDecoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), channels=96,
                 num_classes=1, dropout=0.0, use_wavelet_edges=True,
                 use_reverse_attention=True):
        super().__init__()
        self.use_wavelet_edges = use_wavelet_edges
        self.use_reverse_attention = use_reverse_attention
        self.wavelet = HaarWaveletEdgeHead()
        self.context = MultiScaleContextBlock(encoder_channels[3], channels)
        self.fuse3 = ProgressiveFusionBlock(encoder_channels[2], channels)
        self.fuse2 = ProgressiveFusionBlock(encoder_channels[1], channels)
        self.fuse1 = ProgressiveFusionBlock(encoder_channels[0], channels)
        self.coarse_head = nn.Conv2d(channels, num_classes, 1)
        self.aux_d2_head = nn.Conv2d(channels, num_classes, 1)
        self.boundary_head = nn.Sequential(
            ConvNormAct(channels, channels // 2),
            nn.Conv2d(channels // 2, num_classes, 1),
        )
        self.final_refine = ResidualRefineBlock(channels)
        self.dropout = nn.Dropout2d(dropout)
        self.final_head = nn.Conv2d(channels, num_classes, 3, padding=1)

    @staticmethod
    def resize_logits(logits, output_size):
        return F.interpolate(
            logits, size=(output_size, output_size),
            mode="bilinear", align_corners=False)

    def forward(self, encoder_features, normalized_image, output_size, return_aux=False):
        e1, e2, e3, e4 = encoder_features
        wavelet_edges = (
            self.wavelet(normalized_image) if self.use_wavelet_edges else None)

        d4 = self.context(e4)
        d3 = self.fuse3(e3, d4, wavelet_edges)
        coarse_logits = self.coarse_head(d3)
        reverse_attention = (
            1.0 - torch.sigmoid(coarse_logits.detach())
            if self.use_reverse_attention else None)

        d2 = self.fuse2(e2, d3, wavelet_edges, reverse_attention)
        d1 = self.fuse1(e1, d2, wavelet_edges, reverse_attention)
        final_features = F.interpolate(
            d1, scale_factor=2, mode="bilinear", align_corners=False)
        final_features = self.final_refine(final_features)
        logits = self.final_head(self.dropout(final_features))
        logits = self.resize_logits(logits, output_size)

        if not return_aux:
            return logits
        return {
            "logits": logits,
            "aux_d2": self.resize_logits(self.aux_d2_head(d2), output_size),
            "aux_d3": self.resize_logits(coarse_logits, output_size),
            "boundary_logits": self.resize_logits(
                self.boundary_head(d1), output_size),
        }


class SelectiveFusionBlock(nn.Module):
    """Fuse an encoder skip with the preceding decoder feature."""

    def __init__(self, skip_channels, channels):
        super().__init__()
        self.skip_project = ConvNormAct(
            skip_channels, channels, kernel_size=1, padding=0)
        self.decoder_project = ConvNormAct(
            channels, channels, kernel_size=1, padding=0)
        hidden_channels = max(16, channels // 4)
        self.selective_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )
        self.channel_scale = nn.Parameter(torch.tensor(0.1))
        self.refine = ResidualRefineBlock(channels)

    def forward(self, skip, decoder):
        decoder = F.interpolate(
            decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        decoder = self.decoder_project(decoder)
        skip = self.skip_project(skip)
        gate = self.selective_gate(torch.cat([skip, decoder], dim=1))
        fused = decoder + gate * skip
        fused = fused * (1.0 + self.channel_scale * self.channel_gate(fused))
        return self.refine(fused)


class ReverseResidualCorrection(nn.Module):
    """Correct uncertain foreground regions while retaining an identity path."""

    def __init__(self, channels, initial_scale=0.1, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.correction = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, features, previous_logits):
        reverse = 1.0 - torch.sigmoid(
            previous_logits.detach() / self.temperature)
        reverse = F.interpolate(
            reverse, size=features.shape[-2:], mode="bilinear", align_corners=False)
        return features + self.scale * reverse * self.correction(features)


class CascadeReverseDecoder(nn.Module):
    """P4 coarse-to-fine decoder with stage-local reverse residual correction."""

    def __init__(self, encoder_channels=(64, 128, 256, 512), channels=96,
                 num_classes=1, dropout=0.0):
        super().__init__()
        self.context = MultiScaleContextBlock(encoder_channels[3], channels)
        self.fuse3 = SelectiveFusionBlock(encoder_channels[2], channels)
        self.fuse2 = SelectiveFusionBlock(encoder_channels[1], channels)
        self.fuse1 = SelectiveFusionBlock(encoder_channels[0], channels)
        self.aux_d3_head = nn.Conv2d(channels, num_classes, 1)
        self.aux_d2_head = nn.Conv2d(channels, num_classes, 1)
        self.aux_d1_head = nn.Conv2d(channels, num_classes, 1)
        self.reverse_d2 = ReverseResidualCorrection(channels)
        self.reverse_d1 = ReverseResidualCorrection(channels)
        self.reverse_final = ReverseResidualCorrection(channels)
        self.boundary_head = nn.Sequential(
            ConvNormAct(channels, channels // 2),
            nn.Conv2d(channels // 2, num_classes, 1),
        )
        self.final_refine = ResidualRefineBlock(channels)
        self.dropout = nn.Dropout2d(dropout)
        self.final_head = nn.Conv2d(channels, num_classes, 3, padding=1)

    @staticmethod
    def resize_logits(logits, output_size):
        return F.interpolate(
            logits, size=(output_size, output_size),
            mode="bilinear", align_corners=False)

    def forward(self, encoder_features, output_size, return_aux=False):
        e1, e2, e3, e4 = encoder_features
        d4 = self.context(e4)

        d3 = self.fuse3(e3, d4)
        logits_d3 = self.aux_d3_head(d3)

        d2 = self.fuse2(e2, d3)
        d2 = self.reverse_d2(d2, logits_d3)
        logits_d2 = self.aux_d2_head(d2)

        d1 = self.fuse1(e1, d2)
        d1 = self.reverse_d1(d1, logits_d2)
        logits_d1 = self.aux_d1_head(d1)

        final_features = F.interpolate(
            d1, scale_factor=2, mode="bilinear", align_corners=False)
        final_features = self.reverse_final(final_features, logits_d1)
        final_features = self.final_refine(final_features)
        logits = self.final_head(self.dropout(final_features))
        logits = self.resize_logits(logits, output_size)

        if not return_aux:
            return logits
        return {
            "logits": logits,
            "aux_d1": self.resize_logits(logits_d1, output_size),
            "aux_d2": self.resize_logits(logits_d2, output_size),
            "aux_d3": self.resize_logits(logits_d3, output_size),
            "boundary_logits": self.resize_logits(
                self.boundary_head(d1), output_size),
        }


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class RFBModified(nn.Module):
    """Receptive-field block used by the official SAM2-UNet decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch0 = BasicConv2d(in_channels, out_channels, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, (3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, (5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, (7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branches = [self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)]
        return self.relu(self.conv_cat(torch.cat(branches, dim=1)) + self.conv_res(x))


class SAM2UNetUp(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, decoder, skip):
        decoder = F.interpolate(
            decoder, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([skip, decoder], dim=1))


class AttentionGate(nn.Module):
    """Attention gate on a skip connection (Attention U-Net style).

    Uses the up-sampled decoder signal (gate) to modulate the encoder skip so
    that background/irrelevant responses are suppressed before concatenation.
    """

    def __init__(self, channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or max(1, channels // 2)
        self.Wg = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, skip, gate):
        attention = torch.relu(self.Wg(gate) + self.Wx(skip))
        attention = self.psi(attention)
        return skip * attention


class SAM2UNetUpAttn(nn.Module):
    """SAM2UNetUp with an attention gate applied to the encoder skip."""

    def __init__(self, channels=64):
        super().__init__()
        self.attn = AttentionGate(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, decoder, skip):
        decoder = F.interpolate(
            decoder, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        skip = self.attn(skip, decoder)
        return self.conv(torch.cat([skip, decoder], dim=1))


class MultiScaleDepthwiseDetail(nn.Module):
    """Lightweight local-context block at half input resolution."""

    def __init__(self, channels, dilations=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels, channels, 3, padding=dilation,
                    dilation=dilation, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for dilation in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(dilations), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.fuse(torch.cat([
            branch(inputs) for branch in self.branches
        ], dim=1))


class SAM2UNetDetailRefinement(nn.Module):
    """Half-resolution RGB detail branch that predicts a residual mask correction."""

    def __init__(self, decoder_channels=64, detail_channels=32, num_classes=1,
                 use_multiscale_context=False):
        super().__init__()
        detail_stem_layers = [
            nn.Conv2d(3, detail_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.ReLU(inplace=True),
        ]
        if use_multiscale_context:
            detail_stem_layers.append(MultiScaleDepthwiseDetail(detail_channels))
        else:
            detail_stem_layers.extend([
                nn.Conv2d(
                    detail_channels, detail_channels, 3,
                    padding=1, bias=False),
                nn.BatchNorm2d(detail_channels),
                nn.ReLU(inplace=True),
            ])
        self.detail_stem = nn.Sequential(*detail_stem_layers)
        self.refine = nn.Sequential(
            nn.Conv2d(
                decoder_channels + detail_channels,
                decoder_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Conv2d(decoder_channels, num_classes, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, decoder_feature, normalized_image):
        detail = self.detail_stem(normalized_image)
        decoder_feature = F.interpolate(
            decoder_feature,
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.delta_head(
            self.refine(torch.cat([decoder_feature, detail], dim=1)))


class SAM2UNetDecoder(nn.Module):
    """Official RFB + U-shaped decoder with two deeply supervised outputs."""

    def __init__(self, encoder_channels=(144, 288, 576, 1152), channels=64,
                 num_classes=1, use_detail_refinement=False,
                 use_multiscale_detail=False, use_skip_attention=False, naf=None):
        super().__init__()
        self.use_detail_refinement = use_detail_refinement
        self.rfbs = nn.ModuleList([
            RFBModified(in_channels, channels) for in_channels in encoder_channels
        ])
        up_cls = SAM2UNetUpAttn if use_skip_attention else SAM2UNetUp
        self.up1 = up_cls(channels)
        self.up2 = up_cls(channels)
        self.up3 = up_cls(channels)
        self.side1 = nn.Conv2d(channels, num_classes, 1)
        self.side2 = nn.Conv2d(channels, num_classes, 1)
        self.head = nn.Conv2d(channels, num_classes, 1)
        self.detail_refinement = (
            SAM2UNetDetailRefinement(
                decoder_channels=channels,
                detail_channels=32,
                num_classes=num_classes,
                use_multiscale_context=use_multiscale_detail,
            )
            if use_detail_refinement else None
        )
        if naf == "p12":
            self.frequency_enhancer = PyramidNAF2D(active_stages=(1, 2))
        elif naf == "cascade":
            self.frequency_enhancer = CascadeDiscreteNAF2D(channels=channels)
        elif naf is None:
            self.frequency_enhancer = None
        else:
            raise ValueError("Unsupported NAF variant: {}".format(naf))

    @staticmethod
    def resize_logits(logits, output_size):
        return F.interpolate(
            logits, size=(output_size, output_size),
            mode="bilinear", align_corners=False)

    def forward(self, encoder_features, output_size, return_aux=False,
                normalized_image=None):
        features = [
            rfb(feature) for rfb, feature in zip(self.rfbs, encoder_features)
        ]
        naf_regularization = None
        naf_stats = None
        naf_fusion_stats = None
        if self.frequency_enhancer is not None:
            naf_outputs = self.frequency_enhancer(features)
            features = naf_outputs[0]
            naf_regularization = naf_outputs[1]
            naf_stats = naf_outputs[2]
            if len(naf_outputs) >= 4:
                naf_fusion_stats = naf_outputs[3]
        e1, e2, e3, e4 = features
        d3 = self.up1(e4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up3(d2, e1)
        logits = self.resize_logits(self.head(d1), output_size)
        if self.detail_refinement is not None:
            if normalized_image is None:
                raise ValueError("Detail refinement requires the normalized RGB image")
            delta_logits = self.detail_refinement(d1, normalized_image)
            logits = logits + self.resize_logits(delta_logits, output_size)
        if not return_aux:
            return logits
        output = {
            "logits": logits,
            "aux_d2": self.resize_logits(self.side2(d2), output_size),
            "aux_d3": self.resize_logits(self.side1(d3), output_size),
        }
        if naf_regularization is not None:
            output["naf_regularization"] = naf_regularization
            output["naf_stats"] = naf_stats
            if naf_fusion_stats is not None:
                output["naf_fusion_stats"] = naf_fusion_stats
        return output


class MSLAU_net(nn.Module):

    def __init__(self, img_size=224, mla_channels=64,in_chans=3, num_classes=1,
                 edge_guidance_enabled=True, fusion_mode="fixed", decoder_dropout=0.0,
                 decoder_mode="legacy", progressive_channels=96,
                 p3_use_wavelet_edges=True, p3_use_reverse_attention=True,
                 encoder_name="mslau", hiera_checkpoint=None,
                 freeze_hiera_backbone=True):
        super(MSLAU_net, self).__init__()
        if decoder_mode not in {
                "legacy", "progressive_wavelet", "cascade_reverse",
                "p5_hiera_reverse", "p5_sam2unet",
                "p5_sam2unet_detail", "p5_sam2unet_detail_multiscale",
                "p5_sam2unet_mafa", "p5_sam2unet_mafa_stage",
                "p5_sam2unet_mafa_v2", "p5_sam2unet_mafa_v3",
                "p5_sam2unet_mafa_v4", "p5_sam2unet_mafa_v5",
                "p5_sam2unet_mafa_mr", "p5_sam2unet_mafa_v2gn", "p5_sam2unet_mafa_v2ar", "p5_sam2unet_mafa_v2ag", "p5_sam2unet_mafa_v2_naf", "p5_sam2unet_mafa_v2_nafcascade", "p5_sam2unet_nafcascade"}:
            raise ValueError("Unsupported decoder mode: {}".format(decoder_mode))
        if decoder_mode != "legacy" and fusion_mode != "fixed":
            raise ValueError("Progressive decoders require fusion_mode='fixed'")
        if encoder_name not in {"mslau", "sam2_hiera_large"}:
            raise ValueError("Unsupported encoder: {}".format(encoder_name))
        if decoder_mode in {
                "p5_hiera_reverse", "p5_sam2unet",
                "p5_sam2unet_detail", "p5_sam2unet_detail_multiscale",
                "p5_sam2unet_mafa", "p5_sam2unet_mafa_stage",
                "p5_sam2unet_mafa_v2", "p5_sam2unet_mafa_v3",
                "p5_sam2unet_mafa_v4", "p5_sam2unet_mafa_v5",
                "p5_sam2unet_mafa_mr", "p5_sam2unet_mafa_v2gn", "p5_sam2unet_mafa_v2ar", "p5_sam2unet_mafa_v2ag", "p5_sam2unet_mafa_v2_naf", "p5_sam2unet_mafa_v2_nafcascade", "p5_sam2unet_nafcascade"
        } and encoder_name != "sam2_hiera_large":
            raise ValueError("P5 requires encoder_name='sam2_hiera_large'")
        if encoder_name == "sam2_hiera_large" and decoder_mode not in {
                "p5_hiera_reverse", "p5_sam2unet",
                "p5_sam2unet_detail", "p5_sam2unet_detail_multiscale",
                "p5_sam2unet_mafa", "p5_sam2unet_mafa_stage",
                "p5_sam2unet_mafa_v2", "p5_sam2unet_mafa_v3",
                "p5_sam2unet_mafa_v4", "p5_sam2unet_mafa_v5",
                "p5_sam2unet_mafa_mr", "p5_sam2unet_mafa_v2gn", "p5_sam2unet_mafa_v2ar", "p5_sam2unet_mafa_v2ag", "p5_sam2unet_mafa_v2_naf", "p5_sam2unet_mafa_v2_nafcascade", "p5_sam2unet_nafcascade"}:
            raise ValueError("SAM2 Hiera encoder is currently reserved for P5")
        self.img_size = img_size
        self.norm_cfg = None
        self.mla_channels = mla_channels
        self.BatchNorm = nn.BatchNorm2d
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.decoder_channels = 4 * self.mla_channels
        self.edge_guidance_enabled = edge_guidance_enabled
        self.fusion_mode = fusion_mode
        self.decoder_mode = decoder_mode
        self.encoder_name = encoder_name
        self.decoder_dropout = nn.Dropout2d(p=decoder_dropout)
        self.detail_only_tuning = False

        if encoder_name == "mslau":
            encoder_channels = (64, 128, 256, 512)
            self.encoder = Encoder(
                depth=[4, 8, 11, 5], img_size=img_size, in_chans=3,
                num_classes=1, embed_dim=list(encoder_channels),
                head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None)
        else:
            self.encoder = HieraFeatureEncoder(
                model_name=encoder_name,
                checkpoint_path=hiera_checkpoint,
                freeze_backbone=freeze_hiera_backbone,
                adapter_mode=(
                    "block_mafa_v2ar"
                    if decoder_mode == "p5_sam2unet_mafa_v2ar"
                    else "block_mafa_v2gn"
                    if decoder_mode == "p5_sam2unet_mafa_v2gn"
                    else "block_mafa_mr"
                    if decoder_mode == "p5_sam2unet_mafa_mr"
                    else "block_mafa_v5"
                    if decoder_mode == "p5_sam2unet_mafa_v5"
                    else "block_mafa_v4"
                    if decoder_mode == "p5_sam2unet_mafa_v4"
                    else "block_mafa_v3"
                    if decoder_mode == "p5_sam2unet_mafa_v3"
                    else "block_mafa_v2"
                    if decoder_mode in {
                        "p5_sam2unet_mafa_v2", "p5_sam2unet_mafa_v2ag",
                        "p5_sam2unet_mafa_v2_naf",
                        "p5_sam2unet_mafa_v2_nafcascade"}
                    else "block_mafa_stage"
                    if decoder_mode == "p5_sam2unet_mafa_stage"
                    else "block_mafa" if decoder_mode == "p5_sam2unet_mafa"
                    else "block_prompt" if decoder_mode in {
                        "p5_sam2unet", "p5_sam2unet_detail",
                        "p5_sam2unet_detail_multiscale",
                        "p5_sam2unet_nafcascade"}
                    else "stage_output"),
            )
            encoder_channels = self.encoder.out_channels
        if decoder_mode == "legacy":
            self.conv_mla = Conv_MLA(
                embed_dim=list(encoder_channels), mla_channels=mla_channels,
                fusion_mode=fusion_mode)
            self.mlahead = MLAHead(mla_channels=mla_channels)
            self.edge_guidance = EdgeGuidedAttention(channels=self.decoder_channels)
            self.seg = nn.Conv2d(self.decoder_channels, self.num_classes, 3, padding=1)
            self.progressive_decoder = None
        else:
            self.conv_mla = None
            self.mlahead = None
            self.edge_guidance = None
            self.seg = None
            if decoder_mode in {"progressive_wavelet", "p5_hiera_reverse"}:
                self.progressive_decoder = ProgressiveWaveletDecoder(
                    encoder_channels=encoder_channels,
                    channels=progressive_channels,
                    num_classes=num_classes,
                    dropout=decoder_dropout,
                    use_wavelet_edges=(
                        False if decoder_mode == "p5_hiera_reverse"
                        else p3_use_wavelet_edges),
                    use_reverse_attention=(
                        True if decoder_mode == "p5_hiera_reverse"
                        else p3_use_reverse_attention),
                )
            elif decoder_mode == "cascade_reverse":
                self.progressive_decoder = CascadeReverseDecoder(
                    encoder_channels=encoder_channels,
                    channels=progressive_channels,
                    num_classes=num_classes,
                    dropout=decoder_dropout,
                )
            else:
                naf_kind = None
                if decoder_mode == "p5_sam2unet_mafa_v2_naf":
                    naf_kind = "p12"
                elif decoder_mode == "p5_sam2unet_mafa_v2_nafcascade":
                    naf_kind = "cascade"
                elif decoder_mode == "p5_sam2unet_nafcascade":
                    naf_kind = "cascade"
                self.progressive_decoder = SAM2UNetDecoder(
                    encoder_channels=encoder_channels,
                    channels=64,
                    num_classes=num_classes,
                    use_detail_refinement=(
                        decoder_mode in {
                            "p5_sam2unet_detail",
                            "p5_sam2unet_detail_multiscale"}),
                    use_multiscale_detail=(
                        decoder_mode == "p5_sam2unet_detail_multiscale"),
                    use_skip_attention=(
                        decoder_mode == "p5_sam2unet_mafa_v2ag"),
                    naf=naf_kind,
                )

    def enable_p5_detail_only_tuning(self):
        if self.decoder_mode not in {
                "p5_sam2unet_detail", "p5_sam2unet_detail_multiscale"}:
            raise ValueError(
                "Detail-only tuning requires decoder_mode='p5_sam2unet_detail'")
        self.requires_grad_(False)
        self.progressive_decoder.detail_refinement.requires_grad_(True)
        self.detail_only_tuning = True
        self._keep_p5_base_eval()

    def _keep_p5_base_eval(self):
        if not self.detail_only_tuning:
            return
        self.encoder.eval()
        for name in ("rfbs", "up1", "up2", "up3", "side1", "side2", "head"):
            getattr(self.progressive_decoder, name).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.detail_only_tuning:
            self._keep_p5_base_eval()
            self.progressive_decoder.detail_refinement.train(mode)
        return self

    def forward(self, inputs, return_aux=False, return_features=False):
        edge_inputs = inputs
        if inputs.size()[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        encoder_features = self.encoder(inputs)

        if self.decoder_mode != "legacy":
            if self.decoder_mode in {"progressive_wavelet", "p5_hiera_reverse"}:
                outputs = self.progressive_decoder(
                    encoder_features,
                    normalized_image=edge_inputs,
                    output_size=self.img_size,
                    return_aux=return_aux,
                )
            else:
                outputs = self.progressive_decoder(
                    encoder_features,
                    output_size=self.img_size,
                    return_aux=return_aux,
                    normalized_image=(
                        inputs if self.decoder_mode in {
                            "p5_sam2unet_detail",
                            "p5_sam2unet_detail_multiscale"}
                        else None),
                )
            if return_features:
                if not isinstance(outputs, dict):
                    outputs = {"logits": outputs}
                outputs["encoder_features"] = encoder_features
            return outputs

        conv_mla_features = self.conv_mla(encoder_features)

        decoder_features = self.mlahead(conv_mla_features)
        decoder_features = self.decoder_dropout(decoder_features)
        coarse_logits = self.seg(decoder_features)
        if self.edge_guidance_enabled:
            decoder_features = self.edge_guidance(
                decoder_features,
                coarse_logits.detach(),
                image=edge_inputs,
            )
            logits = self.seg(decoder_features)
        else:
            logits = coarse_logits
        logits = F.interpolate(logits, size=self.img_size, mode='bilinear',
                               align_corners=True)
        if return_features:
            return {"logits": logits, "encoder_features": encoder_features}
        return logits
    
    def load_from(self, pretrained_path=None):
        if pretrained_path:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device,weights_only=False)#weights_only=False
            model_dict = self.encoder.state_dict()
            full_dict = copy.deepcopy(pretrained_dict['model'])
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,model_dict[k].shape))
                        del full_dict[k]
            msg = self.encoder.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")
