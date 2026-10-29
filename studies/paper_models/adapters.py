"""Load pinned author architectures without their data/training/evaluation code.

All edits to author code are explicit AST import/path adaptations, except the
documented FFT dtype guard. Source originals remain untouched and hashed.
"""
import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
ASSETS = WORKSPACE / 'work/paper_models_20261030'


def setup_dependencies():
    path = str(ASSETS / 'deps')
    if path not in sys.path:
        sys.path.insert(0, path)


def source_module(name, path, namespace=None, remove_imports=(), replacements=()):
    text = Path(path).read_text()
    for old, new in replacements:
        if old not in text:
            raise ValueError(f'Expected source adaptation not found in {path}: {old}')
        text = text.replace(old, new)
    tree = ast.parse(text)
    kept = []
    for node in tree.body:
        module = node.module or '' if isinstance(node, ast.ImportFrom) else ''
        if isinstance(node, ast.ImportFrom) and any(module == p or module.startswith(p + '.') for p in remove_imports):
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            continue
        kept.append(node)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(namespace or {})
    sys.modules[name] = module
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


class NoRegistry:
    @staticmethod
    def register_module(*args, **kwargs):
        return lambda cls: cls


class SegmentationHeadBase(nn.Module):
    """Only the author's BaseDecodeHead segmentation layers, without a new loss."""
    def __init__(self, in_channels, channels, num_classes=1, dropout_ratio=.1,
                 input_transform=None, align_corners=False, **kwargs):
        super().__init__()
        self.in_channels, self.channels = in_channels, channels
        self.align_corners = align_corners
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio else None

    def cls_seg(self, feature):
        return self.conv_seg(self.dropout(feature) if self.dropout is not None else feature)


def resize(x, size=None, scale_factor=None, mode='nearest', align_corners=None, **kwargs):
    return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


def make_polyper(pretrained=True):
    setup_dependencies()
    base = ASSETS / 'vendor/polyper/mmsegmentation/mmseg/models'
    embed = source_module('paper_polyper_embed', base / 'utils/embed.py')
    swin = source_module('paper_polyper_swin', base / 'backbones/swin.py',
        dict(MODELS=NoRegistry, PatchEmbed=embed.PatchEmbed, PatchMerging=embed.PatchMerging),
        remove_imports=('mmseg.registry', 'utils.embed'))
    encoder = swin.SwinTransformer(embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, drop_path_rate=.3, patch_norm=True,
        init_cfg=dict(type='Pretrained', checkpoint=str(ASSETS / 'weights/swin_tiny.pth')) if pretrained else None)
    encoder.init_weights()
    decoder = source_module('paper_polyper_head', base / 'decode_heads/polyper.py',
        dict(BaseDecodeHead=SegmentationHeadBase, MODELS=NoRegistry, resize=resize),
        remove_imports=('mmseg.models.decode_heads.decode_head', 'mmseg.registry', 'utils'),
        replacements=(('output.cpu().detach().numpy()', 'output.float().cpu().detach().numpy()'),
                      ('maskd = maskd.cuda()', 'maskd = maskd.to(output.device)'),
                      ('maske = maske.cuda()', 'maske = maske.to(output.device)'),
                      ('z_e = z_e.cuda()', 'z_e = z_e.to(x.device)'),
                      ('z_d = z_d.cuda()', 'z_d = z_d.to(x.device)')))
    head = decoder.Polyper(in_channels=[96, 192, 384, 768], channels=96,
        image_size=128, heads=8, num_classes=1, dropout_ratio=.1)
    from polyper_attention import refine_forward
    for layer in head.modules():
        if isinstance(layer,decoder.RefineAttention):
            layer.forward = types.MethodType(refine_forward,layer)
    # BaseDecodeHead's published init_cfg applies Normal(std=.01) to conv_seg.
    nn.init.normal_(head.conv_seg.weight, std=.01)
    nn.init.zeros_(head.conv_seg.bias)
    return PaperAdapter('polyper', encoder=encoder, decoder=head)


def make_eat(pretrained=True):
    setup_dependencies()
    from dcnv4_reference import DCNv4Function
    base = ASSETS / 'vendor/eat'
    dcn = source_module('paper_dcnv4_module', base / 'DCNv4_op/DCNv4/modules/dcnv4.py',
                        dict(DCNv4Function=DCNv4Function), remove_imports=('functions',))
    package, modules = types.ModuleType('DCNv4'), types.ModuleType('DCNv4.modules')
    modules.DCNv4 = dcn.DCNv4
    package.modules = modules
    sys.modules['DCNv4'], sys.modules['DCNv4.modules'] = package, modules
    module = source_module('paper_eat_network', base / 'src/eat_model/eat.py')
    path = base / 'pretrained/pvt_v2_b2.pth'
    network = module.EAT(gla_os=(3., 1.), L_feature_os=1., model_dir=None)
    if pretrained:
        weights = torch.load(path, map_location='cpu', weights_only=True)
        load_backbone(network.backbone, weights, 'EAT PVTv2-B2')
    return PaperAdapter('eat', network=network)


def load_backbone(model, weights, label):
    weights = weights.get('state_dict', weights.get('model', weights))
    own = model.state_dict()
    selected = {k: v for k, v in weights.items() if k in own and v.shape == own[k].shape}
    missing = set(own) - set(selected)
    # Classification heads do not belong to a feature-only backbone.
    required = {k for k in missing if not k.startswith(('head.', 'fc.'))
                and not k.endswith('.num_batches_tracked')}
    if required:
        raise RuntimeError(f'{label}: missing pretrained backbone keys: {sorted(required)[:12]}')
    model.load_state_dict(selected, strict=False)
    return dict(loaded=len(selected), total=len(own), ignored=sorted(set(weights)-set(selected)))


def make_condseg(pretrained=True):
    setup_dependencies()
    base = ASSETS / 'vendor/condseg/network'
    resnet = source_module('paper_condseg_resnet', base / 'resnet.py')
    def backbone():
        result = resnet.resnet50(pretrained=False)
        if pretrained:
            path = ASSETS / 'weights/resnet50_author.pth'
            if hashlib.sha256(path.read_bytes()).hexdigest() != '19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097':
                raise RuntimeError('Unexpected official legacy ResNet checkpoint')
            # The pinned, hash-verified pytorch.org file predates weights_only
            # serialization support (legacy tar format).
            load_backbone(result, torch.load(path, map_location='cpu', weights_only=False), 'ConDSeg ResNet50')
        return result
    module = source_module('paper_condseg_network', base / 'model.py', dict(resnet50=backbone),
                           remove_imports=('network.resnet',))
    network = module.ConDSeg(H=352, W=352)
    # Preserve exact pre-sigmoid logits rather than applying logit to rounded BF16 probabilities.
    network.output_block.sig = nn.Identity()
    network.aux_head.branch_fg[-1] = nn.Identity()
    network.aux_head.branch_bg[-1] = nn.Identity()
    network.aux_head.branch_uc[-1] = nn.Identity()
    return PaperAdapter('condseg', network=network)


def make_asgnet(pretrained=True):
    setup_dependencies()
    base = ASSETS / 'vendor/asgnet/ASGNet_26_TCSVT/lib'
    pvt = source_module('paper_asgnet_pvt', base / 'pvt_v2.py')
    snp = source_module('paper_asgnet_snp', base / 'module3_NEW_PS.py',
        replacements=(('torch.fft.fft2(x).real', 'torch.fft.fft2(x.float()).real'),))
    edge = source_module('paper_asgnet_edge', base / 'module3_NEW_PS_edge.py')
    namespace = {name: getattr(edge, name) for name in ('MSE', 'DCI_decoder_1', 'DCI_decoder_2', 'DCI_decoder_3', 'DCI_decoder_4')}
    namespace.update(pvt_v2_b3=pvt.pvt_v2_b3, SNP_module=snp.SNP_module)
    module = source_module('paper_asgnet_network', base / 'Network_PVTV3_PS_edge_1_new.py', namespace,
        remove_imports=('lib',), replacements=(("pretrained_dict = torch.load('/pvt_v2_b3.pth')",
        'pretrained_dict = self.shared_encoder.state_dict()'),))
    network = module.Network(channels=96)
    if pretrained:
        load_backbone(network.shared_encoder, torch.load(ASSETS / 'weights/pvt_v2_b3.pth', map_location='cpu', weights_only=True), 'ASGNet PVTv2-B3')
    return PaperAdapter('asgnet', network=network)


class PaperAdapter(nn.Module):
    def __init__(self, name, network=None, encoder=None, decoder=None):
        super().__init__()
        self.name = name
        self.network = network
        self.backbone = encoder
        self.decoder = decoder

    @property
    def encoder(self):
        if self.name == 'polyper':
            return self.backbone
        if self.name == 'eat':
            return self.network.backbone
        if self.name == 'asgnet':
            return self.network.shared_encoder
        # Tuple used only to identify encoder parameters, without registering aliases.
        return (self.network.layer0, self.network.layer1, self.network.layer2, self.network.layer3)

    def forward(self, image, return_aux=False):
        if self.name == 'polyper':
            main = resize(self.decoder(self.backbone(image)), size=image.shape[-2:], mode='bilinear', align_corners=False)
            aux2, aux3 = main, main
        elif self.name == 'eat':
            main = self.network(image)
            aux2, aux3 = main, main
        elif self.name == 'condseg':
            main, aux2, background, _ = self.network(image)
            aux3 = -background
        else:
            values = self.network(image)
            main, aux2, aux3 = values[4], values[3], values[2]
        if main.shape != (image.shape[0], 1, *image.shape[-2:]):
            raise RuntimeError(f'{self.name} prediction shape {main.shape}')
        return dict(logits=main, aux_d2=aux2, aux_d3=aux3) if return_aux else main


FACTORIES = dict(polyper=make_polyper, eat=make_eat, condseg=make_condseg, asgnet=make_asgnet)


def make_model(name, pretrained=True, activation_checkpointing=True):
    from deterministic_ops import replace_global_max_pool
    model = FACTORIES[name](pretrained)
    model.deterministic_global_max_pool_count = replace_global_max_pool(model)
    if name == 'condseg' and activation_checkpointing:
        from checkpointing import checkpoint_condseg
        model.checkpointed_modules = checkpoint_condseg(model.network)
    return model
