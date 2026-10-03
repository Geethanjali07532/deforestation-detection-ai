"""
Module 7: Forest vs Non-Forest Classification
Traditional Machine Learning baseline:
  - Pixel-Level Spectral Feature Extraction
  - Random Forest, SVM, and XGBoost Classifiers
  - Feature Importance Analysis
  - Full-Scene Forest / Non-Forest Mask Inference
"""

from .ml_classifier import PixelFeatureExtractor, ForestMLClassifierSuite

__all__ = ["PixelFeatureExtractor", "ForestMLClassifierSuite"]
