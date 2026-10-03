"""
Module 2: Understanding Satellite Imagery
Handles multispectral satellite imagery loading, metadata inspection,
spectral band extraction (Blue, Green, Red, NIR, SWIR), false-color compositions,
and spectral reflectance signature analysis.
"""

from .spectral_inspector import SatelliteImageryInspector
from .dataset_generator import create_sample_multispectral_geotiff

__all__ = ["SatelliteImageryInspector", "create_sample_multispectral_geotiff"]
