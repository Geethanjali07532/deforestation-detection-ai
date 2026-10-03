"""
run_module_16.py

Master execution script for Module 16: Geospatial Vector Polygon Extraction & GIS Export.
Accomplishes:
  1. Raster-to-vector polygonization of deforestation change masks with Douglas-Peucker simplification
  2. GIS attribute table generation: Alert ID, Area (ha), Perimeter (m), Severity, Driver, Centroids
  3. Multi-format operational GIS export:
     - GeoJSON FeatureCollection (.geojson)
     - ESRI Shapefile bundle (.shp, .shx, .dbf, .prj)
     - Cloud-Optimized GeoTIFF (.tif) with internal 128x128 tiling and pyramids
  4. Publication-grade figures in outputs/module_16/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.enums import Resampling

# Ensure UTF-8 output encoding for Windows command line / PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure local module directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.module_13_severity_classification.severity_indices import DisturbanceSeverityCalculator
from modules.module_14_pattern_analysis.morphological_analyzer import PatchMorphologyExtractor
from modules.module_14_pattern_analysis.driver_classifier import DisturbanceDriverClassifier
from modules.module_16_geospatial_export.vectorizer import RasterPolygonizer
from modules.module_16_geospatial_export.gis_exporter import export_geojson, export_shapefile, export_cog_geotiff


def make_rgb(bands_5ch):
    """Creates normalized RGB composite from (5, H, W) numpy array."""
    def strt(b):
        p2, p98 = np.percentile(b, (2, 98))
        if p98 <= p2:
            return np.clip(b, 0.0, 1.0)
        return np.clip((b - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
    # Red: band 2, Green: band 1, Blue: band 0
    return np.stack([strt(bands_5ch[2]), strt(bands_5ch[1]), strt(bands_5ch[0])], axis=-1)


def main():
    print("\n" + "=" * 80)
    print("🛰️  MODULE 16: GEOSPATIAL VECTOR POLYGON EXTRACTION & GIS EXPORT")
    print("=" * 80)

    # 1. Ingest Test Satellite Scene & Metadata
    print("\n[1/5] Ingesting Georeferenced Satellite Scene...")
    scene_path = "dataset/test/before/scene_test_001.tif"
    after_path = "dataset/test/after/scene_test_001.tif"
    mask_path  = "dataset/test/masks/scene_test_001.tif"

    with rasterio.open(scene_path) as s1:
        scene_t1 = s1.read()
        transform = s1.transform
        crs = s1.crs

    with rasterio.open(after_path) as s2:
        scene_t2 = s2.read()

    with rasterio.open(mask_path) as sm:
        mask = sm.read(1)

    print(f"  • Spatial Dimensions : {mask.shape[0]} x {mask.shape[1]} pixels")
    print(f"  • Coordinate System  : {crs}")
    print(f"  • Affine Transform   :\n    {transform}")

    # 2. Derive Severity and Driver Rasters
    print("\n[2/5] Computing Spectral Disturbance & Driver Rasters...")
    calc = DisturbanceSeverityCalculator()
    extractor = PatchMorphologyExtractor(min_patch_pixels=4)
    driver_clf = DisturbanceDriverClassifier()

    idx_dict = calc.compute_all_indices(scene_t1, scene_t2)
    severity_map = calc.classify_severity_usgs(idx_dict["dnbr"])
    patches, _ = extractor.extract_patch_metrics(mask, dnbr_raster=idx_dict["dnbr"])

    driver_map = np.zeros_like(mask, dtype=np.uint8)
    driver_id_map = {"Road Encroachment": 1, "Wildfire Scar": 2, "Selective Logging": 3, "Agricultural Clearcut": 4}

    import cv2
    for p in patches:
        driver = driver_clf.classify_single_patch(p)
        cnt = p["contour"]
        cv2.drawContours(driver_map, [cnt], -1, driver_id_map[driver], thickness=-1)

    # 3. Vectorize to GeoJSON Features
    print("\n[3/5] Vectorizing Change Detections via Douglas-Peucker Simplification...")
    vectorizer = RasterPolygonizer(min_area_ha=0.05, simplify_tolerance=0.0001, pixel_res_meters=30.0)
    features = vectorizer.polygonize(
        raster_mask=mask,
        transform=transform,
        crs=str(crs),
        severity_map=severity_map,
        driver_map=driver_map,
        dnbr_raster=idx_dict["dnbr"]
    )
    print(f"  • Extracted {len(features)} valid geospatial vector polygons.")

    # 4. Multi-Format GIS Export
    out_dir = "outputs/module_16"
    os.makedirs(out_dir, exist_ok=True)

    print("\n[4/5] Exporting Operational GIS Formats...")
    # 4.1 Export GeoJSON
    geojson_path = os.path.join(out_dir, "deforestation_alerts.geojson")
    export_geojson(features, geojson_path)
    print(f"  💾 [1/3] GeoJSON FeatureCollection -> '{geojson_path}'")

    # 4.2 Export ESRI Shapefile Bundle
    shp_path = os.path.join(out_dir, "shapefile", "deforestation_alerts.shp")
    export_shapefile(features, shp_path)
    print(f"  💾 [2/3] ESRI Shapefile Bundle     -> '{shp_path}' (.shp, .shx, .dbf, .prj)")

    # 4.3 Export Cloud-Optimized GeoTIFF (COG)
    cog_path = os.path.join(out_dir, "deforestation_severity_cog.tif")
    export_cog_geotiff(severity_map, cog_path, transform=transform, crs=str(crs))
    print(f"  💾 [3/3] Cloud-Optimized GeoTIFF   -> '{cog_path}' (Tiled 128x128 + Pyramids)")

    # Print Attribute Table Summary
    records = [f["properties"] for f in features]
    df_attrs = pd.DataFrame(records)
    print("\n" + "=" * 95)
    print("📋 EXPORTED GIS ATTRIBUTE TABLE PREVIEW")
    print("=" * 95)
    print(df_attrs.head(8).to_string(index=False))
    print("=" * 95)

    # Save manifest JSON
    manifest = {
        "total_features": len(features),
        "total_deforested_area_ha": round(float(df_attrs["area_ha"].sum()), 2),
        "crs": str(crs),
        "exported_files": {
            "geojson": geojson_path,
            "shapefile": shp_path,
            "cloud_optimized_geotiff": cog_path
        }
    }
    with open(os.path.join(out_dir, "gis_export_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 5. Generate Visualizations
    print(f"\n[5/5] Generating GIS Cartographic Figures in '{out_dir}/'...")

    # --------------------------------------------------------------------------
    # Figure 1: Vector Polygons Overlaid on Satellite Imagery
    # --------------------------------------------------------------------------
    rgb_t2 = make_rgb(scene_t2)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5), constrained_layout=True)

    ax1.imshow(rgb_t2)
    ax1.set_title("Post-Disturbance (T2) Satellite Image", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(rgb_t2)
    driver_color_hex = {
        "Road Encroachment": "#ffff00",     # Yellow
        "Wildfire Scar": "#ff1744",         # Red
        "Selective Logging": "#00e5ff",     # Cyan
        "Agricultural Clearcut": "#ff9100"  # Orange
    }

    import shapely.geometry
    for feat in features:
        geom = shapely.geometry.shape(feat["geometry"])
        props = feat["properties"]
        c_hex = driver_color_hex.get(props["driver"], "#ffffff")

        # Convert coordinates to pixel space for visualization
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            polys = []

        for p in polys:
            x, y = p.exterior.xy
            # Map geographic coordinates back to pixel coords
            inv_trans = ~transform
            cols, rows = [], []
            for xi, yi in zip(x, y):
                col, row = inv_trans * (xi, yi)
                cols.append(col)
                rows.append(row)

            ax2.plot(cols, rows, color=c_hex, linewidth=2.2)
            ax2.fill(cols, rows, color=c_hex, alpha=0.35)

        # Label polygon ID at centroid
        cx, cy = inv_trans * (props["centroid_x"], props["centroid_y"])
        ax2.text(cx, cy, props["alert_id"].replace("DEF_2026_", "#"),
                 fontsize=8, fontweight="bold", color="black",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=c_hex, alpha=0.85, edgecolor="black"))

    ax2.set_title("Extracted Vector Polygons with Attribution IDs", fontsize=12, fontweight="bold")
    ax2.axis("off")

    legend_handles = [
        mpatches.Patch(color=c, label=d) for d, c in driver_color_hex.items()
    ]
    ax2.legend(handles=legend_handles, loc="lower right", framealpha=0.9, fontsize=9)

    fig1_path = os.path.join(out_dir, "01_vector_polygons_overlay.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  📸 Saved vector polygons overlay -> '{fig1_path}'")

    # --------------------------------------------------------------------------
    # Figure 2: GIS Tabular Attribute Table Summary
    # --------------------------------------------------------------------------
    fig2, ax_table = plt.subplots(figsize=(16, 5), constrained_layout=True)
    ax_table.axis("off")

    display_df = df_attrs[["alert_id", "area_ha", "perimeter_m", "severity", "driver", "mean_dnbr", "status"]].head(10)
    table = ax_table.table(
        cellText=display_df.values,
        colLabels=["Alert ID", "Area (ha)", "Perimeter (m)", "Severity Tier", "Disturbance Driver", "Mean dNBR", "Status"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1565c0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f5f5f5")

    ax_table.set_title("Operational GIS Attribute Table (ESRI Shapefile & GeoJSON DBF Schema)", fontsize=12, fontweight="bold", pad=20)
    fig2_path = os.path.join(out_dir, "02_gis_attribute_table_summary.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  📸 Saved GIS attribute table summary -> '{fig2_path}'")

    # --------------------------------------------------------------------------
    # Figure 3: Cloud-Optimized GeoTIFF Pyramid Structure
    # --------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    with rasterio.open(cog_path) as cog_src:
        ov_factors = [1, 2, 4, 8]
        for idx, factor in enumerate(ov_factors):
            # Read at overview resolution
            out_shape = (cog_src.height // factor, cog_src.width // factor)
            ov_data = cog_src.read(1, out_shape=out_shape, resampling=Resampling.nearest)

            axes3[idx].imshow(ov_data, cmap="YlOrRd", vmin=0, vmax=3)
            axes3[idx].set_title(f"Pyramid Level {idx}\n(Factor {factor}x: {out_shape[1]}x{out_shape[0]})", fontsize=10, fontweight="bold")
            axes3[idx].axis("off")

    fig3_path = os.path.join(out_dir, "03_cog_pyramid_structure.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  📸 Saved COG pyramid visualization -> '{fig3_path}'")

    print("\n" + "=" * 80)
    print("✅ MODULE 16: GEOSPATIAL VECTOR & COG EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
