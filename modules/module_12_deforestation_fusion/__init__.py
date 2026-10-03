"""
Module 12: Deep Learning-Based Deforestation Detection Fusion Strategies

Evaluates 4 Multi-Temporal Fusion Paradigms:
  1. Early Fusion (EF): Input-level channel concatenation (10 channels -> U-Net)
  2. Late Fusion (LF): Dual independent encoders with bottleneck feature concatenation
  3. Siamese Feature Differencing (Siam-Diff): Dual shared encoder with absolute difference skips
  4. Siamese Feature Concatenation (Siam-Concat): Dual shared encoder with concatenated skips
"""

from .fusion_models import EarlyFusionUNet, LateFusionUNet, SiameseDiffUNet, SiameseConcatUNet
from .fusion_comparator import compare_fusion_strategies

__all__ = [
    "EarlyFusionUNet",
    "LateFusionUNet",
    "SiameseDiffUNet",
    "SiameseConcatUNet",
    "compare_fusion_strategies"
]
