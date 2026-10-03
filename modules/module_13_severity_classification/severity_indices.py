"""
severity_indices.py

Module 13: Deforestation Severity & Post-Disturbance Classification
Computes physical disturbance and burn severity indices from pre- and post-disturbance
multispectral imagery (Bands: Blue, Green, Red, NIR, SWIR):
  - Normalized Burn Ratio (NBR)
  - Differenced NBR (dNBR)
  - Relativized dNBR (RdNBR)
  - Relativized Burn Ratio (RBR)
  - Delta Normalized Difference Vegetation Index (dNDVI)
  - Delta Normalized Difference Moisture Index (dNDMI)
"""

import numpy as np
from typing import Dict, Tuple, Optional


class DisturbanceSeverityCalculator:
    """
    Computes spectral disturbance severity indices and maps continuous
    damage gradients into discrete USGS/USFS severity categories.
    """

    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def compute_nbr(self, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """Normalized Burn Ratio: (NIR - SWIR) / (NIR + SWIR)."""
        return (nir - swir) / (nir + swir + self.eps)

    def compute_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
        return (nir - red) / (nir + red + self.eps)

    def compute_ndmi(self, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """Normalized Difference Moisture Index: (NIR - SWIR) / (NIR + SWIR)."""
        return (nir - swir) / (nir + swir + self.eps)

    def compute_all_indices(self, t1_bands: np.ndarray, t2_bands: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculates all pre-, post-, and differential indices from (5, H, W) rasters.
        Bands: 0=Blue, 1=Green, 2=Red, 3=NIR, 4=SWIR.
        """
        # Pre-disturbance (T1)
        red1, nir1, swir1 = t1_bands[2], t1_bands[3], t1_bands[4]
        nbr1 = self.compute_nbr(nir1, swir1)
        ndvi1 = self.compute_ndvi(nir1, red1)
        ndmi1 = self.compute_ndmi(nir1, swir1)

        # Post-disturbance (T2)
        red2, nir2, swir2 = t2_bands[2], t2_bands[3], t2_bands[4]
        nbr2 = self.compute_nbr(nir2, swir2)
        ndvi2 = self.compute_ndvi(nir2, red2)
        ndmi2 = self.compute_ndmi(nir2, swir2)

        # Differenced metrics
        dnbr = nbr1 - nbr2
        dndvi = ndvi1 - ndvi2
        dndmi = ndmi1 - ndmi2

        # Relativized metrics (Miller & Thode 2007; Parks et al. 2014)
        rdnbr = dnbr / np.sqrt(np.abs(nbr1) + 0.001)
        rbr = dnbr / (nbr1 + 1.001)

        return {
            "nbr_pre": nbr1,
            "nbr_post": nbr2,
            "dnbr": dnbr,
            "rdnbr": rdnbr,
            "rbr": rbr,
            "ndvi_pre": ndvi1,
            "ndvi_post": ndvi2,
            "dndvi": dndvi,
            "ndmi_pre": ndmi1,
            "ndmi_post": ndmi2,
            "dndmi": dndmi
        }

    def classify_severity_usgs(self, dnbr: np.ndarray) -> np.ndarray:
        """
        Maps continuous dNBR values to 4 discrete USGS disturbance severity classes:
          0: Undisturbed / Stable (< 0.10)
          1: Low Severity (0.10 - 0.27) [Selective logging / light canopy thinning]
          2: Moderate Severity (0.27 - 0.66) [Heavy thinning / partial crown fire]
          3: High Severity (>= 0.66) [Stand-replacing fire / total clearcut]
        """
        classes = np.zeros_like(dnbr, dtype=np.uint8)
        classes[(dnbr >= 0.10) & (dnbr < 0.27)] = 1
        classes[(dnbr >= 0.27) & (dnbr < 0.66)] = 2
        classes[dnbr >= 0.66] = 3
        return classes

    def compute_severity_area_stats(self, severity_map: np.ndarray, pixel_res_meters: float = 30.0) -> Dict[str, Dict[str, float]]:
        """
        Computes spatial footprint in pixels, hectares, and percentage for each severity tier.
        Assumes pixel resolution in meters (e.g., 30m for Landsat, 10m for Sentinel-2).
        """
        total_pixels = severity_map.size
        pixel_area_ha = (pixel_res_meters * pixel_res_meters) / 10000.0  # 1 ha = 10,000 m^2

        class_names = {
            0: "Undisturbed / Stable",
            1: "Low Severity",
            2: "Moderate Severity",
            3: "High Severity"
        }

        stats = {}
        for c_id, name in class_names.items():
            count = int(np.sum(severity_map == c_id))
            area_ha = count * pixel_area_ha
            pct = (count / total_pixels) * 100.0
            stats[name] = {
                "pixel_count": count,
                "area_hectares": round(area_ha, 2),
                "area_km2": round(area_ha / 100.0, 4),
                "percentage": round(pct, 2)
            }
        return stats
