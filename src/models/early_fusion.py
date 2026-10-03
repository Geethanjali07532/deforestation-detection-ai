"""
src/models/early_fusion.py
Early-Fusion Multi-Class Change Detection Network.
Stacks T1 (5 bands) and T2 (5 bands) -> 10 channels, outputting 4 transition classes:
  0: Forest -> Forest (Intact Forest)
  1: Forest -> Cleared (Mechanical Clearing)
  2: Forest -> Burned (Wildfire / Pyrogenic Burn Scar)
  3: Non-Forest -> Non-Forest (Stable Soil / Background)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .unet import DoubleConv


class EarlyFusion4ClassNet(nn.Module):
    """
    Early Fusion Deep Architecture for 4-Class Land Transition Mapping.
    Inputs: (B, 10, H, W) -> Outputs: (B, 4, H, W) transition class logits.
    """
    CLASS_NAMES = [
        "Forest -> Forest (Intact)",
        "Forest -> Cleared (Logging)",
        "Forest -> Burned (Fire)",
        "Non-Forest -> Non-Forest (Background)"
    ]

    CLASS_COLORS = [
        "#1b5e20",  # Forest -> Forest (Deep Green)
        "#f59e0b",  # Forest -> Cleared (Amber / Orange)
        "#ef4444",  # Forest -> Burned (Crimson Red)
        "#334155"   # Non-Forest -> Non-Forest (Slate Gray)
    ]

    def __init__(self, in_channels: int = 10, num_classes: int = 4, base_features: int = 16):
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

        # 4-class pixel classifier
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Stack 5 bands T1 + 5 bands T2 -> 10 channels
        x = torch.cat([t1, t2], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        xb = self.bot(x4)

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
