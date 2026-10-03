"""
cnn_models.py

Module 8: CNN-Based Land Cover Classification
Implements:
  1. SatelliteLandCoverCNN: Custom multi-stage CNN with feature map extraction.
  2. SatelliteResNet18: Transfer learning backbone adapted for multispectral satellite patches.
"""

import os
import sys
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torchvision.models as models

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ConvBlock(nn.Module):
    """Basic convolutional unit: Conv2D -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class SatelliteLandCoverCNN(nn.Module):
    """
    3-Stage Convolutional Neural Network for satellite land-cover patch classification.
    Supports feature map extraction across layers to inspect edge & texture activations.
    """

    def __init__(self, in_channels: int = 5, num_classes: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stage 1: Low-level edges & spectral color contrasts
        self.stage1 = ConvBlock(in_channels, 32, pool=True)   # 32x32 -> 16x16

        # Stage 2: Mid-level textures (canopy density, furrows, roads)
        self.stage2 = ConvBlock(32, 64, pool=True)            # 16x16 -> 8x8

        # Stage 3: High-level semantic land cover representations
        self.stage3 = ConvBlock(64, 128, pool=True)           # 8x8 -> 4x4

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        out = self.classifier(x)
        return out

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executes forward pass and captures intermediate activation feature maps.
        Returns: (logits, {'stage1': f1, 'stage2': f2, 'stage3': f3})
        """
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        pooled = self.global_pool(f3)
        logits = self.classifier(pooled)
        feature_maps = {
            "Stage 1 (Edges & Color)": f1,
            "Stage 2 (Textures & Boundaries)": f2,
            "Stage 3 (Semantic Land Cover)": f3
        }
        return logits, feature_maps


class SatelliteResNet18(nn.Module):
    """
    Transfer learning ResNet-18 adapted for 5-band multispectral satellite input.
    """

    def __init__(self, in_channels: int = 5, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        # Load backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)

        # Adapt first conv layer to accept in_channels (5 bands)
        orig_conv = base_model.conv1
        self.conv1 = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None
        )

        # Copy original RGB weights for first 3 channels, average for NIR/SWIR
        if pretrained:
            with torch.no_grad():
                self.conv1.weight[:, :3] = orig_conv.weight
                mean_weight = orig_conv.weight.mean(dim=1, keepdim=True)
                self.conv1.weight[:, 3:] = mean_weight.repeat(1, in_channels - 3, 1, 1)

        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
