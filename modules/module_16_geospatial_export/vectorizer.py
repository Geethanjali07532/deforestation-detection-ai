"""
vectorizer.py

Module 16: Geospatial Vector Polygon Extraction & GIS Export
Vectorizes raster change detections into georeferenced polygons with attribute tables:
  - Uses rasterio.features.shapes for affine-projected polygonization
  - Applies Douglas-Peucker geometric simplification via Shapely
  - Filters sub-threshold noise below Minimum Mapping Unit (MMU)
  - Constructs enriched attribute tables (ID, Area ha, Perimeter m, Severity, Driver, Centroids)
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import rasterio
import rasterio.features
import shapely
import shapely.geometry
from shapely.validation import make_valid

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class RasterPolygonizer:
    """
    Transforms raster detection masks into topological vector polygons
    ready for GIS integration.
    """

    def __init__(
        self,
        min_area_ha: float = 0.05,
        simplify_tolerance: float = 0.0001,
        pixel_res_meters: float = 30.0
    ):
        self.min_area_ha = min_area_ha
        self.simplify_tolerance = simplify_tolerance
        self.pixel_res_meters = pixel_res_meters
        self.pixel_area_ha = (pixel_res_meters * pixel_res_meters) / 10000.0

    def polygonize(
        self,
        raster_mask: np.ndarray,
        transform: rasterio.Affine,
        crs: str = "EPSG:4326",
        severity_map: Optional[np.ndarray] = None,
        driver_map: Optional[np.ndarray] = None,
        dnbr_raster: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Converts raster detections into a list of GeoJSON Feature dictionaries.
        """
        mask_binary = (raster_mask > 0).astype(np.uint8)

        # Vectorize using rasterio.features.shapes
        shapes_gen = rasterio.features.shapes(
            mask_binary,
            mask=(mask_binary == 1),
            transform=transform
        )

        features = []
        feat_id = 1

        severity_names = {0: "Undisturbed", 1: "Low Severity", 2: "Moderate", 3: "High Severity"}
        driver_names = {1: "Road Encroachment", 2: "Wildfire Scar", 3: "Selective Logging", 4: "Agricultural Clearcut"}

        for geom_dict, val in shapes_gen:
            if val == 0:
                continue

            raw_geom = shapely.geometry.shape(geom_dict)
            if not raw_geom.is_valid:
                raw_geom = make_valid(raw_geom)

            # Douglas-Peucker simplification
            if self.simplify_tolerance > 0:
                clean_geom = raw_geom.simplify(self.simplify_tolerance, preserve_topology=True)
            else:
                clean_geom = raw_geom

            if clean_geom.is_empty:
                continue

            # Compute area in hectares
            # If coordinates are in lat/lon degrees, approximate area using pixel resolution
            area_px = rasterio.features.geometry_mask([clean_geom], out_shape=raster_mask.shape, transform=transform, invert=True).sum()
            area_ha = area_px * self.pixel_area_ha

            if area_ha < self.min_area_ha:
                continue

            perim_m = clean_geom.length * (111320.0 if "4326" in crs else 1.0)

            # Sample attributes from underlying rasters
            sev_label = "Unassigned"
            driver_label = "Unassigned"
            mean_dnbr = 0.0

            poly_mask = rasterio.features.geometry_mask([clean_geom], out_shape=raster_mask.shape, transform=transform, invert=True)

            if severity_map is not None and np.any(poly_mask):
                sev_vals = severity_map[poly_mask]
                if len(sev_vals) > 0:
                    dom_sev = int(np.bincount(sev_vals).argmax())
                    sev_label = severity_names.get(dom_sev, f"Class {dom_sev}")

            if driver_map is not None and np.any(poly_mask):
                d_vals = driver_map[poly_mask]
                if len(d_vals) > 0:
                    dom_d = int(np.bincount(d_vals).argmax())
                    driver_label = driver_names.get(dom_d, f"Driver {dom_d}")

            if dnbr_raster is not None and np.any(poly_mask):
                d_dnbr = dnbr_raster[poly_mask]
                if len(d_dnbr) > 0:
                    mean_dnbr = float(np.mean(d_dnbr))

            centroid = clean_geom.centroid

            feature = {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(clean_geom),
                "properties": {
                    "alert_id": f"DEF_2026_{feat_id:04d}",
                    "area_ha": round(float(area_ha), 3),
                    "perimeter_m": round(float(perim_m), 1),
                    "severity": sev_label,
                    "driver": driver_label,
                    "mean_dnbr": round(float(mean_dnbr), 3),
                    "centroid_x": round(float(centroid.x), 6),
                    "centroid_y": round(float(centroid.y), 6),
                    "status": "Verified Alert"
                }
            }
            features.append(feature)
            feat_id += 1

        return features
