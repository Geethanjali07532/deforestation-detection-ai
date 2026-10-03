"""
morphological_analyzer.py

Module 14: Fire, Logging & Road Encroachment Pattern Analysis
Extracts quantitative landscape ecology metrics and geometric morphological signatures:
  - PatchMorphologyExtractor: Area, Perimeter, Linearity, Circularity, Solidity, Fractal Dimension
  - ForestFragmentationAnalyzer: Core vs. Edge Forest buffer modeling (100m edge effects)
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, label

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class PatchMorphologyExtractor:
    """
    Extracts geometric, morphological, and spectral attributes from
    individual connected disturbance patches.
    """

    def __init__(self, min_patch_pixels: int = 5, pixel_res_meters: float = 30.0):
        self.min_patch_pixels = min_patch_pixels
        self.pixel_res_meters = pixel_res_meters
        self.pixel_area_ha = (pixel_res_meters * pixel_res_meters) / 10000.0

    def extract_patch_metrics(
        self,
        change_mask: np.ndarray,
        dnbr_raster: Optional[np.ndarray] = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Analyzes all connected components in a binary change mask (1=Deforested, 0=Background).
        Returns:
          - patches_list: List of dictionaries with all computed landscape metrics
          - labeled_mask: 2D integer raster where each pixel has its patch ID
        """
        mask_u8 = (change_mask > 0).astype(np.uint8)
        num_labels, labeled_mask = cv2.connectedComponents(mask_u8)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        patches = []
        patch_id_counter = 1

        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_patch_pixels:
                continue

            perim_px = cv2.arcLength(cnt, closed=True)
            if perim_px <= 0:
                continue

            # 1. Bounding box & Aspect Ratio / Linearity
            rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
            w_box, h_box = rect[1]
            major_axis = max(w_box, h_box)
            minor_axis = max(min(w_box, h_box), 1e-4)
            linearity = float(major_axis / minor_axis)

            # 2. Circularity / Isoperimetric Quotient: 4*pi*A / P^2
            circularity = float((4.0 * np.pi * area_px) / (perim_px * perim_px + 1e-7))

            # 3. Solidity: Area / Convex Hull Area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area_px / (hull_area + 1e-7)) if hull_area > 0 else 1.0

            # 4. Fractal Dimension (Mandelbrot & Lovejoy boundary complexity): 2*ln(P/4)/ln(A)
            if area_px > 1 and perim_px > 4:
                fractal_dim = float(2.0 * np.log(perim_px / 4.0 + 1e-5) / np.log(area_px + 1e-5))
            else:
                fractal_dim = 1.0

            # 5. Centroid
            m = cv2.moments(cnt)
            if m["m00"] > 0:
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
            else:
                cx, cy = rect[0]

            # 6. Mean dNBR (if available)
            mean_dnbr = 0.0
            if dnbr_raster is not None:
                patch_mask_single = np.zeros_like(mask_u8)
                cv2.drawContours(patch_mask_single, [cnt], -1, 1, thickness=-1)
                patch_pixels_dnbr = dnbr_raster[patch_mask_single == 1]
                if len(patch_pixels_dnbr) > 0:
                    mean_dnbr = float(np.mean(patch_pixels_dnbr))

            area_ha = area_px * self.pixel_area_ha
            perim_m = perim_px * self.pixel_res_meters

            patches.append({
                "patch_id": patch_id_counter,
                "area_pixels": int(area_px),
                "area_hectares": round(area_ha, 3),
                "perimeter_pixels": round(float(perim_px), 2),
                "perimeter_meters": round(float(perim_m), 1),
                "linearity": round(linearity, 2),
                "circularity": round(min(1.0, circularity), 4),
                "solidity": round(min(1.0, solidity), 4),
                "fractal_dimension": round(float(fractal_dim), 3),
                "mean_dnbr": round(mean_dnbr, 3),
                "centroid_x": round(cx, 1),
                "centroid_y": round(cy, 1),
                "contour": cnt
            })
            patch_id_counter += 1

        return patches, labeled_mask


class ForestFragmentationAnalyzer:
    """
    Quantifies forest fragmentation, canopy perforation, and edge effects.
    Differentiates between Core Forest (>100m from disturbance) and Edge Forest (<=100m).
    """

    def __init__(self, edge_buffer_meters: float = 100.0, pixel_res_meters: float = 30.0):
        self.edge_buffer_meters = edge_buffer_meters
        self.pixel_res_meters = pixel_res_meters
        self.pixel_area_ha = (pixel_res_meters * pixel_res_meters) / 10000.0

    def analyze_fragmentation(
        self,
        forest_mask: np.ndarray,
        disturbance_mask: np.ndarray
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Computes core forest, edge forest, and degraded edge buffer zones.
        Returns:
          - metrics_dict: Forest intactness and fragmentation statistics
          - fragmentation_map: (0=Non-forest/Deforested, 1=Edge Forest, 2=Core Forest)
        """
        # Remaining forest is forest minus newly disturbed areas
        rem_forest = (forest_mask == 1) & (disturbance_mask == 0)

        # Non-forest boundary including outside world and deforestation
        non_forest = ~rem_forest

        # Distance transform from nearest non-forest edge (in meters)
        dist_from_edge_pixels = distance_transform_edt(rem_forest)
        dist_from_edge_meters = dist_from_edge_pixels * self.pixel_res_meters

        # Core Forest: distance > edge_buffer_meters
        core_forest = rem_forest & (dist_from_edge_meters > self.edge_buffer_meters)

        # Edge Forest: distance <= edge_buffer_meters
        edge_forest = rem_forest & (dist_from_edge_meters <= self.edge_buffer_meters)

        # 3-class fragmentation raster: 0=Disturbed/Non-forest, 1=Edge Forest, 2=Core Forest
        frag_map = np.zeros_like(rem_forest, dtype=np.uint8)
        frag_map[edge_forest] = 1
        frag_map[core_forest] = 2

        # Connected forest patch count
        _, num_patches = label(rem_forest)

        total_rem_ha = np.sum(rem_forest) * self.pixel_area_ha
        core_ha = np.sum(core_forest) * self.pixel_area_ha
        edge_ha = np.sum(edge_forest) * self.pixel_area_ha
        deforested_ha = np.sum(disturbance_mask > 0) * self.pixel_area_ha

        core_pct = (core_ha / (total_rem_ha + 1e-7)) * 100.0
        edge_pct = (edge_ha / (total_rem_ha + 1e-7)) * 100.0
        edge_to_core_ratio = edge_ha / (core_ha + 1e-7)

        metrics = {
            "total_forest_remaining_ha": round(total_rem_ha, 2),
            "core_forest_ha": round(core_ha, 2),
            "core_forest_percentage": round(core_pct, 2),
            "edge_forest_ha": round(edge_ha, 2),
            "edge_forest_percentage": round(edge_pct, 2),
            "edge_to_core_ratio": round(edge_to_core_ratio, 3),
            "new_deforestation_ha": round(deforested_ha, 2),
            "number_of_forest_fragments": int(num_patches),
            "edge_buffer_threshold_meters": self.edge_buffer_meters
        }

        return metrics, frag_map
