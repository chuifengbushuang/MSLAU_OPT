"""Paper-first training recipes; official-code details are identified explicitly."""
import ast
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / 'studies/paper_models'))
import adapters
from checkpointing import checkpoint_module
from deterministic_ops import replace_global_max_pool

ASSETS = adapters.ASSETS
POLICY = {
    'polyper': dict(batch=6, optimizer='AdamW', lr=.0002, weight_decay=.0001,
        betas=[.9, .999], source='AAAI 2024 paper, Implementation Details',
        schedule_source='official main config: 1500-step linear warmup, then linear PolyLR to step160000',
        conflict='Released config lr6e-5/wd.01 is overridden by paper lr2e-4/wd1e-4.',
        loss='single sigmoid cross entropy (BCEWithLogits), official decode_head config',
        augmentation='official RandomResize512 ratio.5..2, crop224, flip.5, PhotoMetricDistortion; final resize352',
        epoch_definition='147 full batches from a continuous shuffled 880-image stream; 882 draws per logical epoch'),
    'eat': dict(batch=8, optimizer='AdamW', lr=.001, weight_decay=.05,
        betas=[.9, .95], source='official code config.yml/train.py; paper full text NOT verified; user accepted this fallback',
        schedule_source='official LinearWarmupCosineAnnealingLR warmup2/max4000, step(epoch) AFTER training',
        loss='MONAI1.2.0 FocalLoss + DiceLoss(smooth_nr=0,smooth_dr=1e-5,sigmoid=True)',
        augmentation='official LAB color transfer within training IDs; Normalize, nearest352, H/Vflip.2',
        conflict='config min_lr1e-7 and weight_decay_end.04 are unused by released train.py; retain executed defaults eta_min0 and fixedwd.05'),
    'condseg': dict(batch=4, optimizer='Adam', lr=.0001, encoder_lr_stage2=.00001,
        weight_decay=0., betas=[.9, .999], source='AAAI2025 paper implementation paragraph and supplement Table6',
        schedule_source='official ReduceLROnPlateau(mode=min,patience30,factor.1), monitored native validation loss',
        loss='stage1 DiceBCE twice + symmetric hard consistency; stage2 DiceBCE(main)+beta_fg*DiceBCE(fg)+beta_bg*DiceBCE(bg)+complementarity',
        augmentation='official phase-specific Rotate/Hflip/Vflip/CoarseDropout; RGB/255 without ImageNet normalization',
        conflict='paper batch4 and stage2 encoderlr1e-5 override public GlaS defaults batch8/12 and encoderlr1e-4',
        stage_budget=dict(maximum_epochs_per_stage=100, patience_per_stage=10, user_confirmed=True),
        adaptation='Declare background as additional mask target. Stage1 commutes linear1x1 projections before bilinear upsampling; BN/ReLU stay at full resolution. Same parameters/math, double-precision forward/gradient/BN oracle checked. Whole-stage activation checkpointing preserves BN/RNG.'),
    'asgnet': dict(batch=40, replica_batch=10, virtual_replicas=4, optimizer='Adam', lr=.0001,
        weight_decay=0., betas=[.9, .999], source='TCSVT2026 paper Experimental Details',
        schedule_source='paper .1 decay every80 epochs; official adjust_lr, 1-based epoch',
        loss='official 5 segmentation Structure heads + 4 raw-output edge Dice heads + 2*cosine_coefficient*UAL',
        augmentation='official data_val.py; edges from supplied edge_util.py distance-transform algorithm, then native dilation',
        conflict='paper step80 schedule overrides extra CosineAnnealingLR in public script; retain official reduce=none legacy BCE behavior and code-only UAL term',
        adaptation='Single GPU sequential 4x10; aggregate mean loss gradients once per global40; retain BatchNorm running buffers only from replica0, like DataParallel4. Activation checkpointing preserves BN/RNG.'),
}


def definitions(path, names, namespace=None):
    """Load exact selected upstream definitions without script side effects."""
    namespace = dict(namespace or {})
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'Missing upstream definitions: {path}: {names}')
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def efficient_stage1_forward(self,image):
    """Linear1x1 projection commutes with bilinear upsampling; BN does not move."""
    features=[]
    value=image
    for index in range(4):
        value=getattr(self,f'layer{index}')(value);features.append(value)
    result=[]
    for index,(feature,scale) in enumerate(zip(features,(2,4,8,16)),1):
        block=getattr(self.head,f'conv{index}')
        projection=block.conv[0]
        if projection.kernel_size!=(1,1) or projection.stride!=(1,1) or projection.padding!=(0,0) or projection.bias is not None:
            raise RuntimeError('Stage1 commutation requires the exact author linear1x1 projection')
        feature=getattr(self,f'up_{scale}x{scale}')(projection(feature))
        feature=block.conv[1](feature)
        result.append(block.relu(feature) if block.act else feature)
    return self.head.sig(self.head.conv_fuse(torch.cat(result,dim=1)))


def make_model(name, stage=2, pretrained=True, checkpointing=True, efficient_stage1=True):
    if name == 'condseg' and stage == 1:
        adapters.setup_dependencies()
        base = ASSETS / 'vendor/condseg/network'
        resnet = adapters.source_module('recipe_stage1_resnet', base/'resnet.py')
        def backbone():
            model = resnet.resnet50(pretrained=False)
            if pretrained:
                path = ASSETS/'weights/resnet50_author.pth'
                if hashlib.sha256(path.read_bytes()).hexdigest() != '19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097':
                    raise RuntimeError('Unexpected ResNet checkpoint')
                adapters.load_backbone(model, torch.load(path, map_location='cpu', weights_only=False), 'stage1 ResNet')
            return model
        module = adapters.source_module('recipe_stage1_model', base/'model_stage1.py',
            dict(resnet50=backbone), remove_imports=('network.resnet',))
        model = adapters.PaperAdapter('condseg', network=module.ConDSegStage1(H=352,W=352))
        replace_global_max_pool(model)
        if efficient_stage1:
            model.network.forward=types.MethodType(efficient_stage1_forward,model.network)
        if checkpointing:
            # Head-level boundaries would retain four FULL-resolution encoder
            # tensors for both views. Checkpoint the whole stage1 graph instead.
            checkpoint_module(model.network)
        return model
    model = adapters.make_model(name, pretrained=pretrained, activation_checkpointing=checkpointing)
    if name == 'condseg':
        model.network.output_block.sig = nn.Sigmoid()
        for key in ('branch_fg','branch_bg','branch_uc'):
            getattr(model.network.aux_head,key)[-1] = nn.Sigmoid()
    if name == 'asgnet' and checkpointing:
        # Non-overlapping top-level modules; no duplicated BN updates on replay.
        for key, module in model.network.named_children():
            if any(True for _ in module.parameters()):
                checkpoint_module(module)
    return model


def forward(model, image, name, stage=2):
    if name == 'polyper':
        return model(image)
    return model.network(image)


def primary(values, name, stage=2):
    if name == 'asgnet':
        return values[4]
    if name == 'condseg' and stage == 2:
        return values[0]
    return values


def probabilities(values, name, stage=2):
    output = primary(values,name,stage)
    return output if name == 'condseg' else output.sigmoid()


def transfer_encoder(source, destination):
    selected = {k:v for k,v in source.state_dict().items()
                if k.startswith(tuple(f'network.layer{i}.' for i in range(4)))}
    expected = {k:v for k,v in destination.state_dict().items() if k in selected}
    if set(selected) != set(expected) or not selected:
        raise RuntimeError('Incomplete stage1 encoder transfer')
    destination.load_state_dict(selected, strict=False)
    if any(not torch.equal(v.detach().cpu(),destination.state_dict()[k].detach().cpu()) for k,v in selected.items()):
        raise RuntimeError('Stage1 encoder transfer mismatch')
    return dict(tensors=len(selected),elements=sum(v.numel() for v in selected.values()))


def optimizer(model, name, stage=2):
    recipe = POLICY[name]
    if name == 'polyper':
        decay, no_decay = [], []
        for key, p in model.named_parameters():
            (no_decay if any(s in key for s in ('absolute_pos_embed','relative_position_bias_table','norm')) else decay).append(p)
        groups = [dict(params=decay,weight_decay=recipe['weight_decay']),dict(params=no_decay,weight_decay=0.)]
    elif name == 'eat':
        # timm create_optimizer_v2 defaults filter_bias_and_bn=True.
        decay, no_decay = [], []
        for key,p in model.named_parameters():
            (no_decay if p.ndim==1 or key.endswith('.bias') else decay).append(p)
        groups = [dict(params=decay,weight_decay=.05),dict(params=no_decay,weight_decay=0.)]
    elif name == 'condseg' and stage == 2:
        enc = [p for key,p in model.named_parameters() if key.startswith(tuple(f'network.layer{i}.' for i in range(4)))]
        ids = {id(p) for p in enc}
        groups = [dict(params=enc,lr=.00001),dict(params=[p for p in model.parameters() if id(p) not in ids],lr=.0001)]
    else:
        groups = model.parameters()
    cls = torch.optim.AdamW if recipe['optimizer']=='AdamW' else torch.optim.Adam
    return cls(groups,lr=recipe['lr'],betas=tuple(recipe['betas']),weight_decay=recipe['weight_decay'])


class Schedule:
    def __init__(self, opt, name):
        self.opt,self.name,self.steps = opt,name,0
        self.native = None
        if name == 'eat':
            module = adapters.source_module('recipe_eat_scheduler',ASSETS/'vendor/eat/src/optimizer.py')
            self.native = module.LinearWarmupCosineAnnealingLR(opt,warmup_epochs=2,max_epochs=4000)
        elif name == 'condseg':
            self.native = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=30)
        elif name == 'polyper':
            adapters.setup_dependencies()
            from mmengine.optim import LinearLR,PolyLR
            self.native = [LinearLR(opt,start_factor=1e-6,by_epoch=False,begin=0,end=1500),
                PolyLR(opt,eta_min=0.,power=1.,begin=1500,end=160000,by_epoch=False)]

    def before_step(self):
        pass  # Native scheduler constructors and after_step own the LR.

    def after_step(self):
        self.steps += 1
        if self.name == 'polyper':
            for scheduler in self.native:
                scheduler.step()

    def before_epoch(self, epoch):
        if self.name == 'asgnet':
            for group in self.opt.param_groups:
                group['lr'] = .0001*(.1**((epoch+1)//80))

    def after_epoch(self, epoch, valid_loss):
        if self.name == 'eat':
            self.native.step(epoch)
        elif self.name == 'condseg':
            self.native.step(valid_loss)

    def state_dict(self):
        state = ([item.state_dict() for item in self.native] if isinstance(self.native,list)
                 else self.native.state_dict() if self.native else None)
        if self.name == 'condseg':
            # ReduceLROnPlateau stores intentional +/-inf sentinels even after
            # its first observation. Encode these in the LOG snapshot only;
            # never mutate or round the scheduler's actual numeric state.
            state = dict(state)
            for key in ('best','mode_worse'):
                value=state[key]
                if math.isinf(value):
                    state[key]='Infinity' if value>0 else '-Infinity'
        return dict(name=self.name,steps=self.steps,native=state)


class NativeLoss:
    def __init__(self, name, stage=2):
        self.name,self.stage = name,stage
        if name == 'eat':
            adapters.setup_dependencies()
            import monai
            if monai.__version__ != '1.2.0':
                raise RuntimeError('EAT requires its released MONAI1.2.0 loss')
            self.focal = monai.losses.FocalLoss(to_onehot_y=False)
            self.dice = monai.losses.DiceLoss(smooth_nr=0,smooth_dr=1e-5,to_onehot_y=False,sigmoid=True)
        if name == 'asgnet':
            path = ASSETS/'vendor/asgnet/ASGNet_26_TCSVT/Train_PS_R_PVTV3_edge1_new1.py'
            self.official = definitions(path, {'structure_loss','dice_loss'}, dict(torch=torch,F=F))
        if name == 'condseg':
            path = ASSETS/'vendor/condseg/utils/metrics.py'
            self.dice_bce = definitions(path,{'DiceBCELoss'},dict(torch=torch,nn=nn,F=F))['DiceBCELoss']()
            self.complementary = definitions(ASSETS/'vendor/condseg/utils/run_engine.py',{'complementary_loss'},dict(torch=torch))['complementary_loss']

    def __call__(self, values, mask, background=None, edge=None, augmented=None, fraction=1.):
        if self.name == 'polyper':
            return F.binary_cross_entropy_with_logits(values,mask)
        if self.name == 'eat':
            return self.focal(values,mask)+self.dice(values,mask)
        if self.name == 'condseg':
            if self.stage == 1:
                loss = self.dice_bce(values,mask)
                if augmented is not None:
                    loss = loss+self.dice_bce(augmented,mask)
                    consistency = F.binary_cross_entropy(values,(augmented>.5).float())
                    consistency = consistency+F.binary_cross_entropy(augmented,(values>.5).float())
                    loss = loss+consistency
                return loss
            main,fg,bg,uc = values
            if background is None:
                raise ValueError('ConDSeg needs aligned native background targets')
            pixels = fg.shape[-2]*fg.shape[-1]
            beta1 = 1/(torch.tanh(fg.sum()/pixels)+1e-15)
            beta2 = 1/(torch.tanh(bg.sum()/pixels)+1e-15)
            pf,pb,pu = torch.stack([fg,bg,uc],dim=1).softmax(dim=1).unbind(dim=1)
            comp = self.complementary(pf,pb,pu)
            return self.dice_bce(main,mask)+beta1*self.dice_bce(fg,mask)+beta2*self.dice_bce(bg,background)+comp
        if edge is None:
            raise ValueError('ASGNet needs edge supervision')
        structure,dice = self.official['structure_loss'],self.official['dice_loss']
        initial = sum(w*structure(v,mask) for w,v in zip((.0625,.125,.25,.5),values[:4]))
        edges = sum(w*dice(v,edge) for w,v in zip((.0625,.125,.25,.5),values[5:]))
        ual = (1-(2*values[4].sigmoid()-1).abs().pow(2)).mean()
        return initial+structure(values[4],mask)+edges+(1-math.cos(fraction*math.pi))*ual


def bn_snapshot(model):
    return [(b,b.detach().clone()) for module in model.modules() if isinstance(module,nn.modules.batchnorm._BatchNorm)
            for b in module.buffers(recurse=False) if b is not None]


def restore_bn(snapshot):
    with torch.no_grad():
        for buffer,value in snapshot:
            buffer.copy_(value)


def backward_asgnet(model, image, mask, edge, criterion, fraction):
    """Emulate author DataParallel4 BN persistence and global40 gradient mean."""
    if len(image) != 40:
        raise ValueError('Native ASGNet global batch must be40')
    initial = bn_snapshot(model)
    retained = None
    loss_sum = 0.
    for replica in range(4):
        if replica:
            restore_bn(initial)
        sl = slice(replica*10,(replica+1)*10)
        outputs = forward(model,image[sl],'asgnet')
        loss = criterion(outputs,mask[sl],edge=edge[sl],fraction=fraction)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite ASGNet loss')
        (loss/4).backward()
        loss_sum += loss.detach().item()/4
        if replica == 0:
            retained = bn_snapshot(model)
    restore_bn(retained)
    return loss_sum
