"""
advanced_models.py

Module 10: Advanced Segmentation Architectures
Implements:
  1. AttentionUNet: Implements Attention Gates (Oktay et al., 2018) on skip connections.
  2. DeepLabV3Lite: Implements Atrous Spatial Pyramid Pooling (ASPP) (Chen et al., 2017).
  3. NestedUNetLite: Implements U-Net++ with dense nested skip connections (Zhou et al., 2018).
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
# 1. ATTENTION U-NET (Attention Gates on Skip Connections)
# ==============================================================================

class AttentionBlock(nn.Module):
    """
    Attention Gate (AG) mechanism:
      - g: Gating signal from deeper decoder stage (contains high-level semantic context).
      - x: Spatial feature map from encoder skip connection (contains high-res spatial detail).
      - Computes attention coefficients alpha in [0, 1] highlighting target forest boundaries
        while suppressing background noise and irrelevant water/cloud areas.
    """

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize g to match spatial size of x if necessary
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode="bilinear", align_corners=False)

        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)  # Attention coefficients (B, 1, H, W)
        return x * alpha, alpha


class AttentionUNet(nn.Module):
    """Attention U-Net adapted for 5-band multispectral satellite segmentation."""

    def __init__(self, in_channels: int = 5, out_channels: int = 1, base_features: int = 16):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DoubleConv(f * 4, f * 8)

        # Attention Gates
        self.att3 = AttentionBlock(f_g=f * 8, f_l=f * 4, f_int=f * 2)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)

        self.att2 = AttentionBlock(f_g=f * 4, f_l=f * 2, f_int=f)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)

        self.att1 = AttentionBlock(f_g=f * 2, f_l=f, f_int=f // 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        # Level 3 with Attention
        att_e3, _ = self.att3(g=b, x=e3)
        d3 = self.up3(b)
        d3 = torch.cat([d3, att_e3], dim=1)
        d3 = self.dec3(d3)

        # Level 2 with Attention
        att_e2, _ = self.att2(g=d3, x=e2)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, att_e2], dim=1)
        d2 = self.dec2(d2)

        # Level 1 with Attention
        att_e1, _ = self.att1(g=d2, x=e1)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, att_e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

    def forward_with_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns logits and attention coefficient maps for visualization."""
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)

        att_e3, a3 = self.att3(g=b, x=e3)
        d3 = self.up3(b)
        d3 = torch.cat([d3, att_e3], dim=1)
        d3 = self.dec3(d3)

        att_e2, a2 = self.att2(g=d3, x=e2)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, att_e2], dim=1)
        d2 = self.dec2(d2)

        att_e1, a1 = self.att1(g=d2, x=e1)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, att_e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.final_conv(d1)
        att_maps = {"Level 1": a1, "Level 2": a2, "Level 3": a3}
        return logits, att_maps


# ==============================================================================
# 2. DEEPLAB V3 LITE (Atrous Spatial Pyramid Pooling - ASPP)
# ==============================================================================

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP):
    Captures multi-scale contextual features using parallel dilated convolutions
    with rates r in [1, 2, 4, 6] plus global image-level pooling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Branch 1: 1x1 conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Branch 2: 3x3 conv with dilation=2
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Branch 3: 3x3 conv with dilation=4
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Branch 4: 3x3 conv with dilation=6
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Branch 5: Image-level global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.b5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[2:]
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        feat5 = F.interpolate(self.b5(self.global_pool(x)), size=(h, w), mode="bilinear", align_corners=False)

        cat = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(cat)


class DeepLabV3Lite(nn.Module):
    """DeepLabV3 architecture with ASPP for satellite land-cover segmentation."""

    def __init__(self, in_channels: int = 5, out_channels: int = 1, base_features: int = 24):
        super().__init__()
        f = base_features

        # Backbone Feature Extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            DoubleConv(f, f * 2),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            DoubleConv(f * 2, f * 4),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            DoubleConv(f * 4, f * 4)  # 16x16
        )

        # ASPP Module
        self.aspp = ASPP(in_channels=f * 4, out_channels=f * 2)

        # Decoder Head
        self.head = nn.Sequential(
            nn.Conv2d(f * 2, f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.size()[2:]
        feat = self.backbone(x)
        aspp_feat = self.aspp(feat)
        out = self.head(aspp_feat)
        # Bilinear upsample back to original input resolution
        return F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)


# ==============================================================================
# 3. U-NET++ (Nested Dense Skip Pathways)
# ==============================================================================

class NestedUNetLite(nn.Module):
    """
    U-Net++ (Nested U-Net) architecture:
    Reduces the semantic gap between encoder and decoder sub-networks
    using dense nested convolutional blocks.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 1, base_features: int = 16):
        super().__init__()
        f = base_features

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Backbone Encoder Nodes
        self.conv0_0 = DoubleConv(in_channels, f)
        self.conv1_0 = DoubleConv(f, f * 2)
        self.conv2_0 = DoubleConv(f * 2, f * 4)

        # Nested Dense Nodes
        self.conv0_1 = DoubleConv(f + f * 2, f)
        self.conv1_1 = DoubleConv(f * 2 + f * 4, f * 2)
        self.conv0_2 = DoubleConv(f * 2 + f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder column 0
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))

        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))

        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        return self.final_conv(x0_2)
