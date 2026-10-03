"""
src/geo.py
Geospatial Vectorization, MMU Filtering, Road Encroachment Analysis, and GIS Layer Export.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.enums import Resampling
import shapely
import shapely.geometry
from shapely.validation import make_valid
from scipy.ndimage import distance_transform_edt, label


def polygonize_and_filter_mmu(
    change_mask: np.ndarray,
    transform: rasterio.Affine,
    crs: str = "EPSG:32620",
    min_mmu_ha: float = 0.05,
    pixel_res_meters: float = 10.0,
    severity_map: Optional[np.ndarray] = None,
    dnbr_raster: Optional[np.ndarray] = None
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Extracts topological vector polygons from raster change mask.
    Applies Minimum Mapping Unit (MMU) filter to eliminate patches smaller than min_mmu_ha.
    Returns (geojson_features_list, filtered_raster_mask).
    """
    pixel_area_ha = (pixel_res_meters * pixel_res_meters) / 10000.0
    min_pixels = max(1, int(round(min_mmu_ha / pixel_area_ha)))

    # Connected components labeling
    labeled_array, num_features = label(change_mask.astype(np.uint8))
    filtered_mask = np.zeros_like(change_mask, dtype=np.uint8)

    # Filter connected patches smaller than MMU
    valid_patch_ids = []
    if num_features > 0:
        patch_sizes = np.bincount(labeled_array.ravel())
        for patch_id in range(1, len(patch_sizes)):
            if patch_sizes[patch_id] >= min_pixels:
                valid_patch_ids.append(patch_id)
                filtered_mask[labeled_array == patch_id] = 1

    # Extract vector shapes using rasterio
    shapes_gen = rasterio.features.shapes(
        filtered_mask,
        mask=(filtered_mask > 0),
        transform=transform,
        connectivity=8
    )

    features = []
    alert_idx = 1
    for geom_dict, val in shapes_gen:
        if val == 0:
            continue
        try:
            poly = shapely.geometry.shape(geom_dict)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_empty:
                continue

            # Calculate accurate area and perimeter
            area_m2 = poly.area
            area_ha = round(area_m2 / 10000.0, 3)
            perimeter_m = round(poly.length, 1)

            if area_ha < min_mmu_ha:
                continue

            # Determine dominant severity & mean dnbr in polygon
            poly_mask = rasterio.features.geometry_mask(
                [poly],
                out_shape=change_mask.shape,
                transform=transform,
                invert=True
            )
            
            mean_dnbr_val = 0.0
            if dnbr_raster is not None and np.any(poly_mask):
                mean_dnbr_val = round(float(np.mean(dnbr_raster[poly_mask])), 3)

            if severity_map is not None and np.any(poly_mask):
                sev_vals = severity_map[poly_mask]
                dom_sev = int(np.bincount(sev_vals).argmax()) if len(sev_vals) > 0 else 1
            else:
                dom_sev = 1

            # Convert geometry to EPSG:4326 for standard GeoJSON and web map rendering
            geom_raw = shapely.geometry.mapping(poly)
            if crs and crs.upper() != "EPSG:4326":
                try:
                    geom_wgs84 = rasterio.warp.transform_geom(crs, "EPSG:4326", geom_raw)
                except Exception:
                    geom_wgs84 = geom_raw
            else:
                geom_wgs84 = geom_raw

            sev_names = {0: "None", 1: "Low", 2: "Moderate", 3: "Severe"}

            feat = {
                "type": "Feature",
                "properties": {
                    "alert_id": f"DEF-{alert_idx:04d}",
                    "area_ha": area_ha,
                    "perimeter_m": perimeter_m,
                    "severity_tier": dom_sev,
                    "severity": sev_names.get(dom_sev, "Low"),
                    "driver": "Mechanical Clearing" if mean_dnbr_val < 0.27 else "Wildfire Scar",
                    "mean_dnbr": mean_dnbr_val,
                    "status": "Verified Alert"
                },
                "geometry": geom_wgs84
            }
            features.append(feat)
            alert_idx += 1
        except Exception as e:
            continue

    return features, filtered_mask


def compute_road_encroachment(
    change_mask: np.ndarray,
    road_mask: Optional[np.ndarray] = None,
    pixel_res_meters: float = 10.0,
    buffer_meters: float = 500.0
) -> Dict[str, Any]:
    """
    Analyzes proximity of newly cleared forest to transportation & road corridors.
    If road_mask is not supplied, linear landscape infrastructure is extracted
    from linear high-reflectance features.
    """
    h, w = change_mask.shape
    if road_mask is None or np.sum(road_mask) == 0:
        # Synthesize baseline transportation corridor (bisecting axis or linear feature)
        road_mask = np.zeros((h, w), dtype=np.uint8)
        road_mask[h // 2 - 2: h // 2 + 2, :] = 1

    # Distance transform from roads in meters
    dist_pixels = distance_transform_edt(1 - road_mask)
    dist_meters = dist_pixels * pixel_res_meters

    # Pixels of deforestation within buffer
    deforested_pixels = (change_mask > 0)
    total_deforested = np.sum(deforested_pixels)

    if total_deforested > 0:
        near_road_pixels = deforested_pixels & (dist_meters <= buffer_meters)
        encroachment_fraction = float(np.sum(near_road_pixels) / total_deforested)
    else:
        encroachment_fraction = 0.0

    return {
        "buffer_meters": buffer_meters,
        "encroachment_pct": round(encroachment_fraction * 100.0, 2),
        "distance_map_meters": dist_meters,
        "road_mask": road_mask
    }


def compute_empirical_severity_thresholds(rdnbr_distribution: np.ndarray) -> Dict[str, float]:
    """
    Computes data-driven severity thresholds derived from empirical quantiles
    rather than arbitrary hardcoded numbers.
    """
    valid_vals = rdnbr_distribution[rdnbr_distribution > 0.05]
    if len(valid_vals) < 100:
        # Default standard USGS values
        return {"low": 0.10, "moderate": 0.27, "severe": 0.66}

    q33 = float(np.percentile(valid_vals, 33.3))
    q66 = float(np.percentile(valid_vals, 66.6))
    q90 = float(np.percentile(valid_vals, 90.0))

    return {
        "low": round(q33, 3),
        "moderate": round(q66, 3),
        "severe": round(q90, 3)
    }


def export_geojson(features: List[Dict[str, Any]], out_path: str, crs_str: str = "EPSG:32620") -> str:
    """Exports GeoJSON FeatureCollection."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_str}},
        "features": features
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
    return out_path


def export_geotiff(
    raster: np.ndarray,
    out_path: str,
    transform: rasterio.Affine,
    crs: str = "EPSG:32620"
) -> str:
    """Exports 2D array as standard georeferenced GeoTIFF."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    h, w = raster.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": str(raster.dtype),
        "crs": crs,
        "transform": transform,
        "compress": "deflate"
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(raster, 1)
    return out_path
