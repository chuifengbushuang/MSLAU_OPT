import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryConfusionMatrix

cfs = BinaryConfusionMatrix()


class DiceLoss_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        inputs = inputs.reshape(inputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        intersection = (inputs * targets).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + smooth)
        return 1 - dice.mean()


class IoU_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        inputs = (inputs > 0.5).float()
        targets = (targets > 0.5).float()

        inputs = inputs.reshape(inputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        intersection = (inputs * targets).sum(dim=1)
        total = (inputs + targets).sum(dim=1)
        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()
