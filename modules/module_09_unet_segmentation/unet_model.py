"""
unet_model.py

Module 9: Forest Segmentation Using U-Net
Full PyTorch implementation of the classic U-Net (Ronneberger et al., 2015)
adapted for 5-band multispectral satellite imagery.

Features:
  - Encoder contracting path capturing contextual representations.
  - Skip connections copying high-resolution spatial feature maps to decoder.
  - Decoder expanding path reconstructing sharp pixel-level forest boundaries.
"""

import os
import sys
import torch
import torch.nn as nn

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
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class ForestUNet(nn.Module):
    """
    U-Net architecture for semantic forest segmentation from satellite imagery.
    Input: (B, in_channels, H, W) where in_channels=5 [B, G, R, NIR, SWIR]
    Output: (B, 1, H, W) raw logits for binary segmentation (Forest=1, Non-Forest=0).
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 1, base_features: int = 16):
        super().__init__()
        f = base_features  # 16 (optimized for fast CPU execution)

        # 1. Encoder (Contracting Path)
        self.enc1 = DoubleConv(in_channels, f)          # (B, 32, H, W)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(f, f * 2)               # (B, 64, H/2, W/2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(f * 2, f * 4)           # (B, 128, H/4, W/4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. Bottleneck
        self.bottleneck = DoubleConv(f * 4, f * 8)     # (B, 256, H/8, W/8)

        # 3. Decoder (Expansive Path with Skip Connections)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)           # (128 up + 128 skip = 256 -> 128)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)           # (64 up + 64 skip = 128 -> 64)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)               # (32 up + 32 skip = 64 -> 32)

        # 4. Final Output Head
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with Skip Connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection from enc3
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection from enc2
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection from enc1
        d1 = self.dec1(d1)

        # Output Logits
        out = self.final_conv(d1)
        return out

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary prediction mask: 1 = Forest, 0 = Non-Forest."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs >= threshold).float()
