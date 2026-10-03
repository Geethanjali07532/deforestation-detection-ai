"""
Module 13: Deforestation Severity & Post-Disturbance Classification

Quantifies disturbance severity using USGS/USFS scientific standards:
  - Differenced Normalized Burn Ratio (dNBR)
  - Relativized dNBR (RdNBR)
  - Relativized Burn Ratio (RBR)
  - Delta Vegetation & Moisture Indices (dNDVI, dNDMI)
  - 4-Tier Severity Categorization:
      Tier 0: Undisturbed / Stable Forest
      Tier 1: Low Severity (Selective Logging / Light Canopy Thinning)
      Tier 2: Moderate Severity (Heavy Thinning / Partial Crown Loss)
      Tier 3: High Severity (Stand-Replacing Fire / Complete Clearcut)
"""

from .severity_indices import DisturbanceSeverityCalculator
from .severity_classifier import ForestDisturbanceClassifier

__all__ = [
    "DisturbanceSeverityCalculator",
    "ForestDisturbanceClassifier"
]
