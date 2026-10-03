"""
gis_exporter.py

Module 16: Geospatial Vector Polygon Extraction & GIS Export
Multi-format geospatial export engine:
  - export_geojson: RFC 7946 Web GIS FeatureCollection
  - export_shapefile: ESRI Shapefile bundle (.shp, .shx, .dbf, .prj) with WGS 84 projection
  - export_cog_geotiff: Cloud-Optimized GeoTIFF with internal tiling and overviews
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
import numpy as np
import rasterio
from rasterio.enums import Resampling
import shapefile

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

WGS84_WKT = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]]'
)


def export_geojson(features: List[Dict[str, Any]], out_path: str, crs_name: str = "urn:ogc:def:crs:OGC:1.3:CRS84") -> str:
    """Exports a list of features as a standard RFC 7946 GeoJSON FeatureCollection."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    fc = {
        "type": "FeatureCollection",
        "name": "deforestation_alerts",
        "crs": {
            "type": "name",
            "properties": {"name": crs_name}
        },
        "features": features
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)

    return out_path


def export_shapefile(features: List[Dict[str, Any]], out_shp_path: str, crs_wkt: str = WGS84_WKT) -> str:
    """
    Exports features as an ESRI Shapefile bundle (.shp, .shx, .dbf, .prj).
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_shp_path)), exist_ok=True)
    base_name = os.path.splitext(out_shp_path)[0]

    with shapefile.Writer(base_name, shapeType=shapefile.POLYGON) as w:
        w.field("ALERT_ID", "C", size=20)
        w.field("AREA_HA", "N", decimal=3)
        w.field("PERIM_M", "N", decimal=1)
        w.field("SEVERITY", "C", size=25)
        w.field("DRIVER", "C", size=30)
        w.field("MEAN_DNBR", "N", decimal=3)
        w.field("CENTROID_X", "N", decimal=6)
        w.field("CENTROID_Y", "N", decimal=6)
        w.field("STATUS", "C", size=20)

        for feat in features:
            geom = feat["geometry"]
            props = feat["properties"]

            # Shapefile expects coordinates as list of rings [[[x, y], ...]]
            if geom["type"] == "Polygon":
                coords = geom["coordinates"]
                w.poly(coords)
            elif geom["type"] == "MultiPolygon":
                # Flatten multi-polygon rings
                rings = []
                for poly in geom["coordinates"]:
                    rings.extend(poly)
                w.poly(rings)
            else:
                continue

            w.record(
                props.get("alert_id", ""),
                props.get("area_ha", 0.0),
                props.get("perimeter_m", 0.0),
                props.get("severity", ""),
                props.get("driver", ""),
                props.get("mean_dnbr", 0.0),
                props.get("centroid_x", 0.0),
                props.get("centroid_y", 0.0),
                props.get("status", "")
            )

    # Write accompanying .prj projection file
    prj_path = base_name + ".prj"
    with open(prj_path, "w", encoding="utf-8") as f_prj:
        f_prj.write(crs_wkt)

    return out_shp_path


def export_cog_geotiff(
    raster_array: np.ndarray,
    out_tif_path: str,
    transform: rasterio.Affine,
    crs: str = "EPSG:4326",
    nodata: Optional[float] = None
) -> str:
    """
    Exports a 2D or 3D numpy array as a Cloud-Optimized GeoTIFF (COG)
    with internal tiling (128x128 blocks), Deflate compression, and pyramids.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_tif_path)), exist_ok=True)

    if raster_array.ndim == 2:
        count = 1
        h, w = raster_array.shape
        data = raster_array[np.newaxis, :, :]
    else:
        count, h, w = raster_array.shape
        data = raster_array

    dtype = str(data.dtype)

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
        "compress": "deflate",
        "nodata": nodata
    }

    with rasterio.open(out_tif_path, "w", **profile) as dst:
        dst.write(data)
        # Build overviews (pyramid layers: 2x, 4x, 8x downsampling)
        dst.build_overviews([2, 4, 8], Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")

    return out_tif_path
