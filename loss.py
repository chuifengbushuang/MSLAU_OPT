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


class BCEDiceLoss_binary(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss_binary()

    def forward(self, inputs, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


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
