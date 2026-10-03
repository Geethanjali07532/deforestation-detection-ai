"""
Module 10: Advanced Segmentation Architectures
Comparative Suite:
  - U-Net (Standard Baseline)
  - Attention U-Net (Attention Gates for Skip Connection Modulation)
  - DeepLabV3-Lite (Atrous Spatial Pyramid Pooling - ASPP)
  - Nested U-Net++ (Dense Nested Skip Pathways)
Benchmarking via IoU (Jaccard Index) and Dice Score.
"""

from .advanced_models import AttentionUNet, DeepLabV3Lite, NestedUNetLite
from .model_comparator import compare_segmentation_models

__all__ = [
    "AttentionUNet",
    "DeepLabV3Lite",
    "NestedUNetLite",
    "compare_segmentation_models"
]
