# All rights reserved.
from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import copy

from networks.attention_factory import build_token_attention
from networks.edge_guidance import EdgeGuidedAttention

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
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_type='msla', attn_kwargs=None):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = build_token_attention(
            attn_type=attn_type,
            dim=dim,
            num_heads=num_heads,
            **(attn_kwargs or {})
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
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H=H, W=W))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), H=H, W=W))
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
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 gfe_attn_type='msla', gfe_crossformer_group_sizes=(7, 7),
                 gfe_crossformer_intervals=(8, 4), gfe_crossformer_adaptive_interval=False,
                 linear_attn_type='legacy'):
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
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer,
                attn_type=gfe_attn_type,
                attn_kwargs=self._build_gfe_attn_kwargs(
                    gfe_attn_type=gfe_attn_type,
                    linear_attn_type=linear_attn_type,
                    stage_index=0,
                    block_index=i,
                    group_sizes=gfe_crossformer_group_sizes,
                    intervals=gfe_crossformer_intervals,
                    adaptive_interval=gfe_crossformer_adaptive_interval,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                ))
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            GFE(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer,
                attn_type=gfe_attn_type,
                attn_kwargs=self._build_gfe_attn_kwargs(
                    gfe_attn_type=gfe_attn_type,
                    linear_attn_type=linear_attn_type,
                    stage_index=1,
                    block_index=i,
                    group_sizes=gfe_crossformer_group_sizes,
                    intervals=gfe_crossformer_intervals,
                    adaptive_interval=gfe_crossformer_adaptive_interval,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                ))
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

    @staticmethod
    def _build_gfe_attn_kwargs(gfe_attn_type, linear_attn_type, stage_index, block_index, group_sizes, intervals,
                               adaptive_interval, qkv_bias, qk_scale, attn_drop, proj_drop):
        if gfe_attn_type.lower() == 'msla':
            return {'linear_attn_type': linear_attn_type}

        if gfe_attn_type.lower() not in {'crossformer', 'crossformer_lsda'}:
            return {}

        return {
            'group_size': group_sizes[stage_index],
            'interval': intervals[stage_index],
            'lsda_flag': block_index % 2,
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'attn_drop': attn_drop,
            'proj_drop': proj_drop,
            'adaptive_interval': adaptive_interval,
        }

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

class Conv_MLA(nn.Module):
    def __init__(self, embed_dim=[64, 128, 256, 512], mla_channels=64, norm_cfg=None):
        super(Conv_MLA, self).__init__()


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

    def forward_branches(self, mla_list):
        head5 = F.interpolate(self.head2(
            mla_list[0]), 2*mla_list[0].shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head3(
            mla_list[1]), 2*mla_list[1].shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head4(
            mla_list[2]), 2*mla_list[2].shape[-1], mode='bilinear', align_corners=True)
        head2 = F.interpolate(self.head5(
            mla_list[3]), 2*mla_list[3].shape[-1], mode='bilinear', align_corners=True)
        return [head5, head4, head3, head2]

    def forward(self, mla_list):
        head5, head4, head3, head2 = self.forward_branches(mla_list)
        return torch.cat([head5, head4, head3, head2], dim=1)

class MSLAU_net(nn.Module):
    OPTIONAL_STATE_PREFIXES = (
        "edge_guidance.",
        "stage_edge_guidance.",
        "stage_edge_pred.",
        "pre_concat_edge_guidance.",
    )
    EDGE_GUIDANCE_POSITIONS = {"none", "decoder", "pre_concat", "stage1", "stage2", "stage3", "stage4"}
    STAGE_CHANNELS = {
        "stage1": 64,
        "stage2": 128,
        "stage3": 256,
        "stage4": 512,
    }

    def __init__(self, img_size=224, mla_channels=64,in_chans=3, num_classes=1,
                 gfe_attn_type='msla', gfe_crossformer_group_sizes=(7, 7),
                 gfe_crossformer_intervals=(8, 4), gfe_crossformer_adaptive_interval=False,
                 edge_guidance_enabled=False, linear_attn_type='legacy',
                 edge_guidance_position=None, edge_guidance_positions=None):
        super(MSLAU_net, self).__init__()
        self.img_size = img_size
        self.norm_cfg = None
        self.mla_channels = mla_channels
        self.BatchNorm = nn.BatchNorm2d
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.encoder_in_chans = 3
        self.decoder_channels = 4 * self.mla_channels
        self.edge_guidance_positions = self._resolve_edge_guidance_positions(
            edge_guidance_enabled=edge_guidance_enabled,
            edge_guidance_position=edge_guidance_position,
            edge_guidance_positions=edge_guidance_positions,
        )
        self.edge_guidance_position = self.edge_guidance_positions[0] if self.edge_guidance_positions else "none"
        self.edge_guidance_enabled = bool(self.edge_guidance_positions)

        self.encoder = Encoder(
            depth=[4, 8, 11, 5], img_size=img_size, in_chans=self.encoder_in_chans, num_classes=1, embed_dim=[64, 128, 256, 512],
            head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            gfe_attn_type=gfe_attn_type,
            gfe_crossformer_group_sizes=gfe_crossformer_group_sizes,
            gfe_crossformer_intervals=gfe_crossformer_intervals,
            gfe_crossformer_adaptive_interval=gfe_crossformer_adaptive_interval,
            linear_attn_type=linear_attn_type)
        self.conv_mla = Conv_MLA(embed_dim=[64, 128, 256, 512], mla_channels=64)
        self.mlahead = MLAHead(mla_channels=64)
        self.edge_guidance = EdgeGuidedAttention(channels=self.decoder_channels)
        self.pre_concat_edge_guidance = nn.ModuleList(
            [
                EdgeGuidedAttention(channels=self.mla_channels)
                for _ in range(4)
            ] if "pre_concat" in self.edge_guidance_positions else []
        )
        stage_positions = [position for position in self.edge_guidance_positions if position in self.STAGE_CHANNELS]
        self.stage_edge_guidance = nn.ModuleDict(
            {
                position: EdgeGuidedAttention(channels=self.STAGE_CHANNELS[position])
                for position in stage_positions
            }
        )
        self.stage_edge_pred = nn.ModuleDict(
            {
                position: nn.Conv2d(self.STAGE_CHANNELS[position], self.num_classes, kernel_size=1)
                for position in stage_positions
            }
        )
        self.seg = nn.Conv2d(self.decoder_channels, self.num_classes, 3, padding=1)

    @classmethod
    def _parse_edge_guidance_positions(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            raw_positions = [item.strip().lower() for item in value.split(",") if item.strip()]
        else:
            raw_positions = [str(item).strip().lower() for item in value if str(item).strip()]

        positions = []
        for position in raw_positions:
            if position not in cls.EDGE_GUIDANCE_POSITIONS:
                raise ValueError(f"Unsupported edge guidance position: {position}")
            if position == "none":
                if len(raw_positions) > 1:
                    raise ValueError("'none' cannot be combined with other edge guidance positions.")
                return tuple()
            if position not in positions:
                positions.append(position)
        return tuple(positions)

    @classmethod
    def _resolve_edge_guidance_positions(cls, edge_guidance_enabled, edge_guidance_position, edge_guidance_positions):
        parsed_positions = cls._parse_edge_guidance_positions(edge_guidance_positions)
        if parsed_positions is not None:
            return parsed_positions

        parsed_position = cls._parse_edge_guidance_positions(edge_guidance_position)
        if parsed_position is not None:
            return parsed_position

        return ("decoder",) if edge_guidance_enabled else tuple()

    @staticmethod
    def _strip_legacy_cem_keys(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict

        stale_prefixes = ("encoder.patch_embed1_cem.", "stem.")
        return OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if not key.startswith(stale_prefixes)
        )

    def _inject_optional_state(self, state_dict):
        current_state = super().state_dict()
        compatible_state = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if key in current_state or not key.startswith(self.OPTIONAL_STATE_PREFIXES)
        )
        for key, value in current_state.items():
            if key in compatible_state:
                continue
            if key.startswith(self.OPTIONAL_STATE_PREFIXES):
                compatible_state[key] = value
        return compatible_state

    def load_state_dict(self, state_dict, strict=True):
        compatible_state = self._strip_legacy_cem_keys(state_dict)
        compatible_state = self._inject_optional_state(compatible_state)
        return super().load_state_dict(compatible_state, strict=strict)

    def _apply_stage_edge_guidance(self, stage_name, feature, image):
        if stage_name not in self.edge_guidance_positions:
            return feature
        stage_logits = self.stage_edge_pred[stage_name](feature)
        return self.stage_edge_guidance[stage_name](
            feature,
            stage_logits.detach(),
            image=image,
        )

    def _apply_pre_concat_edge_guidance(self, branches, coarse_logits, image):
        if "pre_concat" not in self.edge_guidance_positions:
            return branches

        return [
            edge_guidance(branch, coarse_logits.detach(), image=image)
            for edge_guidance, branch in zip(self.pre_concat_edge_guidance, branches)
        ]

    def _encode_with_optional_stage_guidance(self, encoder_inputs, edge_inputs):
        if not any(position in self.STAGE_CHANNELS for position in self.edge_guidance_positions):
            return self.encoder(encoder_inputs)

        features = []
        x = self.encoder.patch_embed1(encoder_inputs)
        x = self.encoder.pos_drop(x)
        for blk in self.encoder.blocks1:
            x = blk(x)
        x = self._apply_stage_edge_guidance("stage1", x, edge_inputs)
        features.append(x)

        x = self.encoder.patch_embed2(x)
        for blk in self.encoder.blocks2:
            x = blk(x)
        x = self._apply_stage_edge_guidance("stage2", x, edge_inputs)
        features.append(x)

        x = self.encoder.patch_embed3(x)
        for blk in self.encoder.blocks3:
            x = blk(x)
        x = self._apply_stage_edge_guidance("stage3", x, edge_inputs)
        features.append(x)

        x = self.encoder.patch_embed4(x)
        for blk in self.encoder.blocks4:
            x = blk(x)
        x = self._apply_stage_edge_guidance("stage4", x, edge_inputs)
        features.append(x)
        return features

    def forward(self, inputs):
        edge_inputs = inputs
        encoder_inputs = inputs
        if encoder_inputs.size()[1] == 1:
            encoder_inputs = encoder_inputs.repeat(1, self.encoder_in_chans, 1, 1)
        encoder_features = self._encode_with_optional_stage_guidance(encoder_inputs, edge_inputs)

        conv_mla_features = self.conv_mla(encoder_features)

        if "pre_concat" in self.edge_guidance_positions:
            decoder_branches = self.mlahead.forward_branches(conv_mla_features)
            coarse_decoder_features = torch.cat(decoder_branches, dim=1)
            coarse_logits = self.seg(coarse_decoder_features)
            decoder_branches = self._apply_pre_concat_edge_guidance(
                decoder_branches,
                coarse_logits,
                edge_inputs,
            )
            decoder_features = torch.cat(decoder_branches, dim=1)
        else:
            decoder_features = self.mlahead(conv_mla_features)
        coarse_logits = self.seg(decoder_features)
        if "decoder" in self.edge_guidance_positions:
            # Detach the coarse prediction so the guidance path stays stable.
            decoder_features = self.edge_guidance(
                decoder_features,
                coarse_logits.detach(),
                image=edge_inputs,
            )
        logits = self.seg(decoder_features)
        logits = F.interpolate(logits, size=self.img_size, mode='bilinear',
                               align_corners=True)
        return logits
    
    def load_from(self):
        pretrained_path = './pretrained/best.pth'#预训练模型路径
        if pretrained_path is not None:
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
    
    
    
