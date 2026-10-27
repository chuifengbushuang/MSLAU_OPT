"""The existing discrete NAF cascade with three zero-initialized coarse skips.

D4=P4; Di=NAFi(Pi, D{i+1}) + beta_i*Up(D{i+1}). The original NAFi
already includes the local Pi identity path. No additional Pi or UpBlock.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CrossScaleSkipNAF2D(nn.Module):
    def __init__(self, original):
        super().__init__()
        # Keep all original parameter names and tensors; zeros consume no RNG.
        self.stages = original.stages
        self.betas = nn.ParameterDict({f'p{i}': nn.Parameter(torch.zeros(())) for i in (1, 2, 3)})

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
            filtered, penalty, stats, gate = self.stages[f'p{level}'](features[level - 1], context)
            with torch.autocast(device_type=filtered.device.type, enabled=False):
                work_dtype = torch.float64 if filtered.dtype == torch.float64 else torch.float32
                coarse = F.interpolate(context.to(work_dtype), size=filtered.shape[-2:],
                                       mode='bilinear', align_corners=True)
                context = (filtered.to(work_dtype) + self.betas[f'p{level}'] * coarse).to(filtered.dtype)
            enhanced[level - 1] = context
            penalties[level], statistics[level], fusion[level] = penalty, stats, gate
        regularizer = torch.stack([penalties[i] for i in (1, 2, 3)]).sum() / 6.0
        return (enhanced, regularizer, torch.stack([statistics[i] for i in (1, 2, 3)]),
                torch.stack([fusion[i] for i in (1, 2, 3)]))
