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
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("BCE and Dice weights must be non-negative and cannot both be zero.")
        weight_sum = bce_weight + dice_weight
        self.bce_weight = bce_weight / weight_sum
        self.dice_weight = dice_weight / weight_sum
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss_binary()

    def forward(self, inputs, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        return self.bce_weight * self.bce(inputs, targets) + self.dice_weight * self.dice(inputs, targets)

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
