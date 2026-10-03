"""
losses_metrics.py

Module 9: Forest Segmentation Using U-Net
Implements:
  1. Dice Loss: Handles extreme spatial class imbalances.
  2. Combined BCE + Dice Loss: Optimal compound objective for semantic segmentation.
  3. IoU (Intersection over Union / Jaccard Index) & Dice Score metrics.
"""

import os
import sys
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class DiceLoss(nn.Module):
    """
    Computes soft Dice loss for binary segmentation:
      Dice = 2 * |P ∩ G| / (|P| + |G|)
      Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        denominator = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice


class CombinedBCEDiceLoss(nn.Module):
    """
    Hybrid loss function:
      Total Loss = alpha * BCE_With_Logits + (1 - alpha) * Dice_Loss
    Combines smooth pixel-level gradient propagation from BCE with global
    overlap optimization from Dice loss.
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


def compute_iou_dice(
    preds_binary: torch.Tensor,
    targets_binary: torch.Tensor,
    eps: float = 1e-6
) -> Tuple[float, float]:
    """
    Calculates batch-averaged Intersection-over-Union (IoU) and Dice coefficient.
    Input tensors should be binary (0 or 1).
    """
    p = preds_binary.view(-1)
    g = targets_binary.view(-1)

    intersection = (p * g).sum().item()
    total_p = p.sum().item()
    total_g = g.sum().item()

    union = total_p + total_g - intersection
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (total_p + total_g + eps)

    return float(iou), float(dice)
