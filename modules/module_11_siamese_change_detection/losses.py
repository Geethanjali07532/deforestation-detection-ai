"""
losses.py

Module 11: Multi-Temporal Change Detection Using Siamese Neural Networks
Loss functions and performance evaluation metrics tailored for sparse change detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DiceLoss(nn.Module):
    """Computes Dice loss for binary segmentation / change detection."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class CompoundChangeLoss(nn.Module):
    """
    Hybrid loss combining Weighted Binary Cross Entropy and Dice Loss.
    Handles class imbalance in deforestation masks.
    """

    def __init__(self, alpha: float = 0.5, pos_weight: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Match device for pos_weight
        if self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)

        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.alpha * loss_bce + (1.0 - self.alpha) * loss_dice


def compute_change_metrics(preds_bin: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    Computes Precision, Recall, Dice/F1, and IoU for binary change detection.
    Inputs should be binary tensors (0 or 1).
    """
    preds_flat = preds_bin.view(-1).long()
    targets_flat = targets.view(-1).long()

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1 - targets_flat)).sum().item()
    fn = ((1 - preds_flat) * targets_flat).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = (2 * precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return float(precision), float(recall), float(dice), float(iou)
