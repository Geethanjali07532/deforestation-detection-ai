"""
fusion_models.py

Module 12: Deep Learning-Based Deforestation Detection Fusion Strategies
Implements 4 distinct multi-temporal fusion paradigms:
  1. EarlyFusionUNet: Input-level temporal band stacking (10 channels -> U-Net)
  2. LateFusionUNet: Dual independent encoders with bottleneck feature concatenation
  3. SiameseDiffUNet: Shared-weight encoder with absolute difference skip connections
  4. SiameseConcatUNet: Shared-weight encoder with concatenated skip connections
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


# ==============================================================================
# 1. EARLY FUSION (Input Concatenation: 10 channels -> U-Net)
# ==============================================================================

class EarlyFusionUNet(nn.Module):
    """
    Early Fusion: Concatenates pre-disturbance (T1) and post-disturbance (T2)
    bands directly at the input stage along the spectral channel dimension (10 bands).
    """

    def __init__(self, in_channels_per_image: int = 5, out_channels: int = 1, base_features: int = 16):
        super().__init__()
        f = base_features
        total_in_channels = in_channels_per_image * 2  # 5 + 5 = 10 bands

        # Encoder
        self.enc1 = DoubleConv(total_in_channels, f)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConv(f * 4, f * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Concatenate along channel dimension: (B, 10, H, W)
        x = torch.cat([t1, t2], dim=1)

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)


# ==============================================================================
# 2. LATE FUSION (Dual Independent Encoders -> Bottleneck Fusion)
# ==============================================================================

class LateFusionUNet(nn.Module):
    """
    Late Fusion: Two completely independent encoders extract representations
    for T1 and T2 respectively, fusing only at the bottleneck layer.
    """

    def __init__(self, in_channels_per_image: int = 5, out_channels: int = 1, base_features: int = 16):
        super().__init__()
        f = base_features

        # Branch 1 (T1 Encoder)
        self.enc1_t1 = DoubleConv(in_channels_per_image, f)
        self.pool1_t1 = nn.MaxPool2d(2, 2)
        self.enc2_t1 = DoubleConv(f, f * 2)
        self.pool2_t1 = nn.MaxPool2d(2, 2)
        self.enc3_t1 = DoubleConv(f * 2, f * 4)
        self.pool3_t1 = nn.MaxPool2d(2, 2)
        self.b_t1 = DoubleConv(f * 4, f * 4)

        # Branch 2 (T2 Encoder - Independent Weights)
        self.enc1_t2 = DoubleConv(in_channels_per_image, f)
        self.pool1_t2 = nn.MaxPool2d(2, 2)
        self.enc2_t2 = DoubleConv(f, f * 2)
        self.pool2_t2 = nn.MaxPool2d(2, 2)
        self.enc3_t2 = DoubleConv(f * 2, f * 4)
        self.pool3_t2 = nn.MaxPool2d(2, 2)
        self.b_t2 = DoubleConv(f * 4, f * 4)

        # Fused Bottleneck Projection
        self.fuse_bottleneck = DoubleConv(f * 8, f * 8)

        # Decoder Head
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 4, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 2, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Branch 1 (T1)
        p1_1 = self.pool1_t1(self.enc1_t1(t1))
        p2_1 = self.pool2_t1(self.enc2_t1(p1_1))
        p3_1 = self.pool3_t1(self.enc3_t1(p2_1))
        feat_t1 = self.b_t1(p3_1)

        # Branch 2 (T2)
        p1_2 = self.pool1_t2(self.enc1_t2(t2))
        p2_2 = self.pool2_t2(self.enc2_t2(p1_2))
        p3_2 = self.pool3_t2(self.enc3_t2(p2_2))
        feat_t2 = self.b_t2(p3_2)

        # Bottleneck concatenation and synthesis
        fused = self.fuse_bottleneck(torch.cat([feat_t1, feat_t2], dim=1))

        d3 = self.dec3(self.up3(fused))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))

        return self.final_conv(d1)


# ==============================================================================
# 3. SIAMESE DIFFERENCING (Shared Encoder + Absolute Difference Skips)
# ==============================================================================

class SiameseDiffUNet(nn.Module):
    """
    Siamese Differencing: Shared-weight encoder with absolute differential
    skip connections D_k = |f_1^k - f_2^k| passed to the decoder.
    """

    def __init__(self, in_channels_per_image: int = 5, out_channels: int = 1, base_features: int = 16):
        super().__init__()
        f = base_features

        # Shared Encoder
        self.enc1 = DoubleConv(in_channels_per_image, f)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConv(f * 4, f * 8)

        # Decoder Head
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        return e1, e2, e3, b

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        e1_1, e2_1, e3_1, b_1 = self._encode(t1)
        e1_2, e2_2, e3_2, b_2 = self._encode(t2)

        diff_b = torch.abs(b_1 - b_2)
        diff_3 = torch.abs(e3_1 - e3_2)
        diff_2 = torch.abs(e2_1 - e2_2)
        diff_1 = torch.abs(e1_1 - e1_2)

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


# ==============================================================================
# 4. SIAMESE CONCATENATION (Shared Encoder + Concatenated Skips)
# ==============================================================================

class SiameseConcatUNet(nn.Module):
    """
    Siamese Concatenation: Shared-weight encoder, but skip connections are
    concatenated [f_1^k; f_2^k] letting the decoder learn arbitrary non-linear differences.
    """

    def __init__(self, in_channels_per_image: int = 5, out_channels: int = 1, base_features: int = 16):
        super().__init__()
        f = base_features

        # Shared Encoder
        self.enc1 = DoubleConv(in_channels_per_image, f)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConv(f * 4, f * 8)

        # Bottleneck projection: 16f -> 8f
        self.b_project = nn.Conv2d(f * 16, f * 8, kernel_size=1)

        # Decoder Head: concatenates upsampled features with [e1; e2]
        # Level 3: up3(8f -> 4f) + (4f + 4f = 8f) = 12f
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 12, f * 4)

        # Level 2: up2(4f -> 2f) + (2f + 2f = 4f) = 6f
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 6, f * 2)

        # Level 1: up1(2f -> f) + (f + f = 2f) = 3f
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 3, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        return e1, e2, e3, b

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        e1_1, e2_1, e3_1, b_1 = self._encode(t1)
        e1_2, e2_2, e3_2, b_2 = self._encode(t2)

        cat_b = self.b_project(torch.cat([b_1, b_2], dim=1))
        cat_3 = torch.cat([e3_1, e3_2], dim=1)
        cat_2 = torch.cat([e2_1, e2_2], dim=1)
        cat_1 = torch.cat([e1_1, e1_2], dim=1)

        d3 = self.up3(cat_b)
        d3 = torch.cat([d3, cat_3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, cat_2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, cat_1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)
