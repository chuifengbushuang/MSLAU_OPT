import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class StructureLoss_binary(nn.Module):
    """Boundary-aware weighted BCE + weighted IoU from SAM2-UNet/PraNet."""

    def __init__(self, kernel_size=31, boundary_weight=5.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.boundary_weight = boundary_weight

    def forward(self, inputs, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        padding = self.kernel_size // 2
        weights = 1.0 + self.boundary_weight * torch.abs(
            F.avg_pool2d(
                targets, kernel_size=self.kernel_size,
                stride=1, padding=padding) - targets)
        weighted_bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none")
        weighted_bce = (weights * weighted_bce).sum(dim=(2, 3)) / (
            weights.sum(dim=(2, 3)).clamp_min(1e-7))

        probabilities = torch.sigmoid(inputs)
        intersection = (probabilities * targets * weights).sum(dim=(2, 3))
        union = ((probabilities + targets) * weights).sum(dim=(2, 3))
        weighted_iou = 1.0 - (intersection + 1.0) / (
            union - intersection + 1.0)
        return (weighted_bce + weighted_iou).mean()


def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension with respect to sorted errors."""
    num_pixels = len(gt_sorted)
    positives = gt_sorted.sum()
    intersection = positives - gt_sorted.float().cumsum(0)
    union = positives + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-7)
    if num_pixels > 1:
        jaccard = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
    return jaccard


class LovaszHingeLoss_binary(nn.Module):
    """Per-image binary Lovasz hinge loss operating directly on logits."""

    def forward(self, inputs, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        losses = []
        for logits, labels in zip(inputs, targets):
            logits = logits.reshape(-1)
            labels = labels.reshape(-1)
            signs = 2.0 * labels - 1.0
            errors = 1.0 - logits * signs
            errors_sorted, permutation = torch.sort(errors, descending=True)
            labels_sorted = labels[permutation]
            losses.append(torch.dot(torch.relu(errors_sorted), lovasz_grad(labels_sorted)))
        return torch.stack(losses).mean()


class BCEDiceLovaszLoss_binary(nn.Module):
    """Weighted BCE + Dice + Lovasz loss for final-mask supervision."""

    def __init__(self, bce_weight=0.35, dice_weight=0.35, lovasz_weight=0.30):
        super().__init__()
        weights = (bce_weight, dice_weight, lovasz_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Loss weights must be non-negative and cannot all be zero.")
        weight_sum = sum(weights)
        self.bce_weight = bce_weight / weight_sum
        self.dice_weight = dice_weight / weight_sum
        self.lovasz_weight = lovasz_weight / weight_sum
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss_binary()
        self.lovasz = LovaszHingeLoss_binary()

    def forward(self, inputs, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        return (
            self.bce_weight * self.bce(inputs, targets)
            + self.dice_weight * self.dice(inputs, targets)
            + self.lovasz_weight * self.lovasz(inputs, targets)
        )

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
