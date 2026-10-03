"""
Module 5: Vegetation Index Analysis
Comprehensive suite of remote sensing spectral indices:
  - NDVI: Normalized Difference Vegetation Index
  - EVI:  Enhanced Vegetation Index
  - SAVI: Soil-Adjusted Vegetation Index
  - NDWI: Normalized Difference Water/Moisture Index
  - NBR:  Normalized Burn Ratio
Includes automated vegetation thresholding and classification.
"""

from .vegetation_indices import VegetationIndexCalculator, VegetationClassifier

__all__ = ["VegetationIndexCalculator", "VegetationClassifier"]
