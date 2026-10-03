"""
src/models/siamese.py
Siamese Neural Network Architecture for Multi-Temporal Deforestation Change Detection.
Dual-stream weight-sharing encoder with multi-scale feature differencing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .unet import DoubleConv


class SiameseChangeEncoder(nn.Module):
    """Weight-sharing multi-scale feature extraction branch."""
    def __init__(self, in_channels: int = 5, base_features: int = 16):
        super().__init__()
        f = base_features
        self.conv1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f * 8, f * 16)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        cb = self.bottleneck(self.pool4(c4))
        return c1, c2, c3, c4, cb


class SiameseChangeDetector(nn.Module):
    """
    Siamese Change Detection Network.
    Passes T1 and T2 through the shared encoder, computes multi-scale absolute feature
    differences |F1 - F2|, and decodes into a change probability mask.
    """
    def __init__(self, in_channels: int = 5, base_features: int = 16):
        super().__init__()
        f = base_features
        # Shared Encoder
        self.encoder = SiameseChangeEncoder(in_channels=in_channels, base_features=f)

        # Decoder with Skip Connections from Feature Differences
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 16, f * 8)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 8, f * 4)

        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 4, f * 2)

        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(f * 2, f)

        # Output head (binary change logits)
        self.classifier = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Weight-sharing feature extraction
        f1_1, f1_2, f1_3, f1_4, f1_b = self.encoder(t1)
        f2_1, f2_2, f2_3, f2_4, f2_b = self.encoder(t2)

        # Multi-scale absolute feature differences
        diff_1 = torch.abs(f1_1 - f2_1)
        diff_2 = torch.abs(f1_2 - f2_2)
        diff_3 = torch.abs(f1_3 - f2_3)
        diff_4 = torch.abs(f1_4 - f2_4)
        diff_b = torch.abs(f1_b - f2_b)

        # Decoder
        d1 = self.up1(diff_b)
        d1 = self.dec1(torch.cat([d1, diff_4], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, diff_3], dim=1))

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, diff_2], dim=1))

        d4 = self.up4(d3)
        d4 = self.dec4(torch.cat([d4, diff_1], dim=1))

        logits = self.classifier(d4)
        return logits
