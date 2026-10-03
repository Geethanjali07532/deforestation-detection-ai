"""
src/models/unet.py
U-Net Deep Semantic Segmentation Architecture for Satellite Forest / Non-Forest Delineation.
Equipped with encoder-decoder skip connections and compound BCE + Dice loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForestUNet(nn.Module):
    """
    Standard U-Net architecture adapted for 5-band multi-spectral satellite imagery.
    Inputs: (B, 5, H, W) -> Outputs: (B, 1, H, W) logits.
    """
    def __init__(self, in_channels: int = 5, num_classes: int = 1, base_features: int = 16):
        super().__init__()
        f = base_features

        # Encoder
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))

        # Bottleneck
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(f * 16, f * 8)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(f * 8, f * 4)

        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(f * 4, f * 2)

        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(f * 2, f)

        # Output projection head
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        xb = self.bot(x4)

        # Decoder with skip concatenations
        d1 = self.up1(xb)
        d1 = self.conv1(torch.cat([d1, x4], dim=1))

        d2 = self.up2(d1)
        d2 = self.conv2(torch.cat([d2, x3], dim=1))

        d3 = self.up3(d2)
        d3 = self.conv3(torch.cat([d3, x2], dim=1))

        d4 = self.up4(d3)
        d4 = self.conv4(torch.cat([d4, x1], dim=1))

        logits = self.outc(d4)
        return logits


class DiceBCELoss(nn.Module):
    """Compound BCE + Soft Dice Loss for handling extreme spatial imbalance."""
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
