"""
vegetation_indices.py

Module 5: Vegetation Index Analysis
Implements the core biophysical remote sensing indices:
  1. NDVI: Normalized Difference Vegetation Index
  2. EVI:  Enhanced Vegetation Index (atmospheric and soil background corrected)
  3. SAVI: Soil Adjusted Vegetation Index
  4. NDWI: Normalized Difference Water/Moisture Index (Gao & McFeeters)
  5. NBR:  Normalized Burn Ratio (fire and canopy destruction)

Provides land-cover thresholding and automated vegetation mapping.
"""

import os
import sys
from typing import Dict, Tuple, Optional, Union
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class VegetationIndexCalculator:
    """
    Computes spectral vegetation, moisture, and burn indices from multispectral rasters.
    Input raster expected shape: (5, H, W)
      Band 0: Blue  (490 nm)
      Band 1: Green (560 nm)
      Band 2: Red   (665 nm)
      Band 3: NIR   (842 nm)
      Band 4: SWIR  (1610 nm)
    """

    def __init__(self, bands: np.ndarray, eps: float = 1e-7):
        if bands.ndim != 3 or bands.shape[0] < 4:
            raise ValueError(f"Expected 3D array with at least 4 bands, got shape {bands.shape}")
        self.bands = bands.astype(np.float32)
        self.eps = eps

        self.blue = self.bands[0]
        self.green = self.bands[1]
        self.red = self.bands[2]
        self.nir = self.bands[3]
        self.swir = self.bands[4] if bands.shape[0] >= 5 else None

    def calculate_ndvi(self) -> np.ndarray:
        """
        Normalized Difference Vegetation Index (NDVI)
        Formula: (NIR - Red) / (NIR + Red)
        Range: [-1.0, 1.0]
        - High positive values (> 0.6): Dense tropical forest canopy.
        - Moderate values (0.2 - 0.5): Sparse or degraded vegetation.
        - Low values (0.0 - 0.2): Bare dry soil, roads, and clearings.
        - Negative values (< 0): Water bodies, rivers, clouds.
        """
        numerator = self.nir - self.red
        denominator = self.nir + self.red + self.eps
        ndvi = numerator / denominator
        return np.clip(ndvi, -1.0, 1.0)

    def calculate_evi(self, G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> np.ndarray:
        """
        Enhanced Vegetation Index (EVI)
        Formula: G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)
        Optimized for dense tropical rainforests where NDVI saturates (at leaf area index > 3).
        Corrects for atmospheric aerosols using the Blue band and canopy background.
        """
        numerator = self.nir - self.red
        denominator = self.nir + (C1 * self.red) - (C2 * self.blue) + L + self.eps
        evi = G * (numerator / denominator)
        return np.clip(evi, -1.0, 1.5)

    def calculate_savi(self, L: float = 0.5) -> np.ndarray:
        """
        Soil Adjusted Vegetation Index (SAVI)
        Formula: ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        Mitigates soil brightness influences in newly cleared patches, open savannas,
        and early-stage deforestation fronts.
        """
        numerator = self.nir - self.red
        denominator = self.nir + self.red + L + self.eps
        savi = (numerator / denominator) * (1.0 + L)
        return np.clip(savi, -1.0, 1.0)

    def calculate_ndwi_moisture(self) -> np.ndarray:
        """
        Gao's Normalized Difference Water/Moisture Index (NDWI)
        Formula: (NIR - SWIR) / (NIR + SWIR)
        Directly measures canopy liquid water thickness.
        High in healthy moist rainforest; plummets when trees are felled and dry out.
        """
        if self.swir is None:
            raise ValueError("SWIR band required to compute NDWI Moisture index.")
        numerator = self.nir - self.swir
        denominator = self.nir + self.swir + self.eps
        ndwi = numerator / denominator
        return np.clip(ndwi, -1.0, 1.0)

    def calculate_ndwi_water(self) -> np.ndarray:
        """
        McFeeters' Normalized Difference Water Index (NDWI_water)
        Formula: (Green - NIR) / (Green + NIR)
        Delineates open surface water bodies (e.g. rivers) from terrestrial features.
        """
        numerator = self.green - self.nir
        denominator = self.green + self.nir + self.eps
        ndwi_w = numerator / denominator
        return np.clip(ndwi_w, -1.0, 1.0)

    def calculate_nbr(self) -> np.ndarray:
        """
        Normalized Burn Ratio (NBR)
        Formula: (NIR - SWIR) / (NIR + SWIR)
        Crucial for detecting forest fires and burned slash clearings (Module 14).
        Healthy forest canopy: High NIR + Low SWIR => High positive NBR.
        Burned/charred scar: Low NIR + High SWIR => Strongly negative NBR.
        """
        if self.swir is None:
            raise ValueError("SWIR band required to compute NBR.")
        numerator = self.nir - self.swir
        denominator = self.nir + self.swir + self.eps
        nbr = numerator / denominator
        return np.clip(nbr, -1.0, 1.0)

    def compute_all_indices(self) -> Dict[str, np.ndarray]:
        """Calculates and returns a dictionary of all 5 indices."""
        indices = {
            "NDVI": self.calculate_ndvi(),
            "EVI": self.calculate_evi(),
            "SAVI": self.calculate_savi(),
            "NDWI_Moisture": self.calculate_ndwi_moisture(),
            "NDWI_Water": self.calculate_ndwi_water(),
            "NBR": self.calculate_nbr()
        }
        return indices


class VegetationClassifier:
    """
    Classifies satellite imagery based on biophysical vegetation index thresholds.
    Distinguishes Forest vs. Non-Forest, Water, Bare Soil, and Degraded vegetation.
    """

    def __init__(self, forest_threshold: float = 0.50):
        self.forest_threshold = forest_threshold

    def create_vegetation_map(
        self,
        ndvi: np.ndarray,
        water_thresh: float = 0.0,
        soil_thresh: float = 0.25,
        degraded_thresh: float = 0.55
    ) -> np.ndarray:
        """
        Generates 4-class discrete Vegetation & Land Cover Map:
          0: Water / Shadow (< water_thresh)
          1: Cleared / Bare Soil (water_thresh to soil_thresh)
          2: Degraded / Sparse Vegetation (soil_thresh to degraded_thresh)
          3: Dense Forest Canopy (>= degraded_thresh)
        """
        veg_map = np.zeros_like(ndvi, dtype=np.uint8)
        veg_map[ndvi < water_thresh] = 0
        veg_map[(ndvi >= water_thresh) & (ndvi < soil_thresh)] = 1
        veg_map[(ndvi >= soil_thresh) & (ndvi < degraded_thresh)] = 2
        veg_map[ndvi >= degraded_thresh] = 3
        return veg_map

    def binary_forest_mask(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Generates binary forest classification mask:
          1: Forest (NDVI >= threshold)
          0: Non-Forest (NDVI < threshold)
        """
        return (ndvi >= self.forest_threshold).astype(np.uint8)

    def compute_class_statistics(self, veg_map: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculates area and percentage breakdown for each land cover class."""
        total_pixels = veg_map.size
        classes = {
            0: "Water / Shadow",
            1: "Cleared / Bare Soil",
            2: "Degraded / Edge Forest",
            3: "Dense Forest Canopy"
        }
        stats = {}
        for class_id, label in classes.items():
            count = int(np.sum(veg_map == class_id))
            pct = (count / total_pixels) * 100.0
            # 10m x 10m pixel = 0.01 hectares
            area_ha = count * 0.01
            stats[label] = {
                "pixel_count": count,
                "percentage": round(pct, 2),
                "area_ha": round(area_ha, 2)
            }
        return stats
