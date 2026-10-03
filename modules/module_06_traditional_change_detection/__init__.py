"""
Module 6: Traditional Forest Change Detection
Implements baseline non-deep-learning change detection algorithms:
  - Spectral Image Differencing
  - NDVI Differencing: (NDVI_before - NDVI_after)
  - Statistical & Empirical Threshold-based Detection
  - Morphological False-Positive Filtering
  - Rigorous Pixel-Level Evaluation (Precision, Recall, F1, IoU, FPR, FNR)
"""

from .change_detector import TraditionalChangeDetector, ChangeEvaluator

__all__ = ["TraditionalChangeDetector", "ChangeEvaluator"]
