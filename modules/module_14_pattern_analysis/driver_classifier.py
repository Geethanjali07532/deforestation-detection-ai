"""
driver_classifier.py

Module 14: Fire, Logging & Road Encroachment Pattern Analysis
Classifies spatial disturbance patches into 4 primary ecological drivers:
  1. Road Encroachment / Linear Corridors (High Linearity, Low Circularity)
  2. Wildfire Scars (High dNBR, High Fractal Dimension, Low Solidity)
  3. Selective Logging / Canopy Gaps (Small Area, Isolated Punctures)
  4. Large-Scale Agricultural Clearcuts (Large Area, High Solidity, Geometric)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


class DisturbanceDriverClassifier:
    """
    Morphological and spectral rule-based & statistical driver classifier.
    Assigns each detected deforestation patch to its underlying anthropological or natural cause.
    """

    DRIVERS = [
        "Road Encroachment",
        "Wildfire Scar",
        "Selective Logging",
        "Agricultural Clearcut"
    ]

    DRIVER_COLORS = {
        "Road Encroachment": "#fbc02d",    # Bright Yellow
        "Wildfire Scar": "#d32f2f",        # Crimson Red
        "Selective Logging": "#00bcd4",    # Cyan
        "Agricultural Clearcut": "#ff6f00" # Deep Orange
    }

    def classify_single_patch(self, p: Dict[str, Any]) -> str:
        """Classifies a single patch dictionary based on geometric & spectral signatures."""
        area_ha = p.get("area_hectares", 0.0)
        linearity = p.get("linearity", 1.0)
        circularity = p.get("circularity", 1.0)
        solidity = p.get("solidity", 1.0)
        fractal_dim = p.get("fractal_dimension", 1.0)
        mean_dnbr = p.get("mean_dnbr", 0.0)

        # 1. Road Encroachment / Linear Corridor: High aspect ratio / high elongation
        if linearity >= 3.0 or (linearity >= 2.2 and circularity < 0.25):
            return "Road Encroachment"

        # 2. Wildfire Scar: Prominent spectral burn signal + irregular fractal perimeter
        if mean_dnbr >= 0.40 and (fractal_dim >= 1.25 or solidity < 0.65):
            return "Wildfire Scar"

        # 3. Small Selective Logging Gaps
        if area_ha < 2.0 and linearity < 2.5:
            return "Selective Logging"

        # 4. Large Agricultural / Industrial Clearcuts
        if area_ha >= 4.0 and solidity >= 0.65:
            return "Agricultural Clearcut"

        # Fallback based on dominant geometry
        if linearity > 2.0:
            return "Road Encroachment"
        elif mean_dnbr > 0.35:
            return "Wildfire Scar"
        elif area_ha > 3.0:
            return "Agricultural Clearcut"
        else:
            return "Selective Logging"

    def classify_all_patches(self, patches: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Classifies all extracted patches and returns a structured DataFrame
        with driver labels and summary metrics.
        """
        records = []
        for p in patches:
            driver = self.classify_single_patch(p)
            rec = {
                "patch_id": p["patch_id"],
                "driver": driver,
                "area_hectares": p["area_hectares"],
                "area_pixels": p["area_pixels"],
                "perimeter_meters": p["perimeter_meters"],
                "linearity": p["linearity"],
                "circularity": p["circularity"],
                "solidity": p["solidity"],
                "fractal_dimension": p["fractal_dimension"],
                "mean_dnbr": p["mean_dnbr"],
                "centroid_x": p["centroid_x"],
                "centroid_y": p["centroid_y"]
            }
            records.append(rec)

        df = pd.DataFrame(records)
        return df

    def compute_driver_statistics(self, df_patches: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Computes total area, patch count, and percentage share by disturbance driver."""
        stats = {}
        total_area_ha = df_patches["area_hectares"].sum()
        total_patches = len(df_patches)

        for driver in self.DRIVERS:
            sub = df_patches[df_patches["driver"] == driver]
            count = len(sub)
            area_ha = sub["area_hectares"].sum() if count > 0 else 0.0
            mean_patch_ha = sub["area_hectares"].mean() if count > 0 else 0.0
            pct_area = (area_ha / (total_area_ha + 1e-7)) * 100.0

            stats[driver] = {
                "patch_count": count,
                "total_area_ha": round(area_ha, 2),
                "mean_patch_ha": round(mean_patch_ha, 2),
                "area_percentage": round(pct_area, 2),
                "color": self.DRIVER_COLORS[driver]
            }

        return stats
