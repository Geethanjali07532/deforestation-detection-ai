"""
siamese_model.py

Module 11: Multi-Temporal Change Detection Using Siamese Neural Networks
Implements SiameseUNetChangeDetector:
  - Dual-branch shared-weight encoder (E_theta) for T1 (pre) and T2 (post) imagery
  - Multi-scale absolute feature differencing at every skip resolution
  - Change decoder reconstructing high-resolution binary deforestation change masks
"""

import os
import sys
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SiameseEncoder(nn.Module):
    """Shared-weight feature encoder processing individual temporal satellite scenes."""

    def __init__(self, in_channels: int = 5, base_features: int = 16):
        super().__init__()
        f = base_features
        self.enc1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConv(f * 4, f * 8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)
        return e1, e2, e3, b


class SiameseUNetChangeDetector(nn.Module):
    """
    Siamese U-Net Change Detection Network:
    - Encodes T1 and T2 through shared SiameseEncoder
    - Computes multi-scale differential features: D_k = |f_1^k - f_2^k|
    - Decodes difference features to yield pixel-accurate deforestation probability
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 1, base_features: int = 16):
        super().__init__()
        f = base_features

        # Shared Siamese Encoder
        self.encoder = SiameseEncoder(in_channels=in_channels, base_features=f)

        # Decoder Head
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # 1. Shared Feature Extraction
        e1_t1, e2_t1, e3_t1, b_t1 = self.encoder(t1)
        e1_t2, e2_t2, e3_t2, b_t2 = self.encoder(t2)

        # 2. Multi-Scale Feature Differencing
        diff_b = torch.abs(b_t1 - b_t2)
        diff_3 = torch.abs(e3_t1 - e3_t2)
        diff_2 = torch.abs(e2_t1 - e2_t2)
        diff_1 = torch.abs(e1_t1 - e1_t2)

        # 3. Progressive Decoder Synthesis
        d3 = self.up3(diff_b)
        d3 = torch.cat([d3, diff_3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, diff_2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, diff_1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

    def forward_with_features(self, t1: torch.Tensor, t2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns change logits alongside feature differential maps for interpretability."""
        e1_t1, e2_t1, e3_t1, b_t1 = self.encoder(t1)
        e1_t2, e2_t2, e3_t2, b_t2 = self.encoder(t2)

        diff_b = torch.abs(b_t1 - b_t2)
        diff_3 = torch.abs(e3_t1 - e3_t2)
        diff_2 = torch.abs(e2_t1 - e2_t2)
        diff_1 = torch.abs(e1_t1 - e1_t2)

        d3 = self.up3(diff_b)
        d3 = torch.cat([d3, diff_3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, diff_2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, diff_1], dim=1)
        d1 = self.dec1(d1)

        logits = self.final_conv(d1)

        features = {
            "diff_level_1": diff_1.mean(dim=1, keepdim=True),
            "diff_level_2": diff_2.mean(dim=1, keepdim=True),
            "diff_level_3": diff_3.mean(dim=1, keepdim=True),
            "diff_bottleneck": diff_b.mean(dim=1, keepdim=True)
        }
        return logits, features
