"""
Module 14: Fire, Logging & Road Encroachment Pattern Analysis

Landscape Ecology & Spatial Pattern Recognition:
  - PatchMorphologyExtractor: Extracts geometric metrics (Area, Perimeter, Linearity, Circularity, Solidity, Fractal Dimension)
  - ForestFragmentationAnalyzer: Computes core vs. edge forest buffer zones (100m edge effects)
  - DisturbanceDriverClassifier: Classifies disturbance drivers (Road Encroachment, Wildfire Scars, Selective Logging, Clearcuts)
"""

from .morphological_analyzer import PatchMorphologyExtractor, ForestFragmentationAnalyzer
from .driver_classifier import DisturbanceDriverClassifier

__all__ = [
    "PatchMorphologyExtractor",
    "ForestFragmentationAnalyzer",
    "DisturbanceDriverClassifier"
]
