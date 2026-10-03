"""
run_module_14.py

Master execution script for Module 14: Fire, Logging & Road Encroachment Pattern Analysis.
Accomplishes:
  1. Multi-temporal satellite imagery ingestion & change mask extraction
  2. Quantitative landscape ecology & morphological metrics:
     - Area, Perimeter, Linearity, Circularity, Solidity, Fractal Dimension
  3. Disturbance driver classification:
     - Road Encroachment / Linear Corridors
     - Wildfire Scars
     - Selective Logging / Canopy Gaps
     - Agricultural Clearcuts
  4. Forest fragmentation & 100m edge effect buffer analysis (Core vs. Edge Forest)
  5. Publication-quality figures in outputs/module_14/
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import cv2

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
from modules.module_14_pattern_analysis.morphological_analyzer import (
    PatchMorphologyExtractor, ForestFragmentationAnalyzer
)
from modules.module_14_pattern_analysis.driver_classifier import DisturbanceDriverClassifier


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
    print("🛰️  MODULE 14: FIRE, LOGGING & ROAD ENCROACHMENT PATTERN ANALYSIS")
    print("=" * 80)

    # 1. Initialize Analytics Engines
    print("\n[1/5] Initializing Landscape Ecology & Morphological Analyzers...")
    calc = DisturbanceSeverityCalculator()
    extractor = PatchMorphologyExtractor(min_patch_pixels=4, pixel_res_meters=30.0)
    driver_clf = DisturbanceDriverClassifier()
    frag_analyzer = ForestFragmentationAnalyzer(edge_buffer_meters=100.0, pixel_res_meters=30.0)

    # 2. Ingest Test Scenes
    print("\n[2/5] Ingesting Multi-Temporal Satellite Imagery & Masks...")
    test_before = sorted(glob.glob("dataset/test/before/*.tif"))
    test_after  = sorted(glob.glob("dataset/test/after/*.tif"))
    test_masks  = sorted(glob.glob("dataset/test/masks/*.tif"))

    all_patches = []
    scenes_data = []

    for b_path, a_path, m_path in zip(test_before, test_after, test_masks):
        with rasterio.open(b_path) as s1:
            t1 = s1.read()
        with rasterio.open(a_path) as s2:
            t2 = s2.read()
        with rasterio.open(m_path) as sm:
            mask = sm.read(1)

        idx = calc.compute_all_indices(t1, t2)
        patches, labeled_mask = extractor.extract_patch_metrics(mask, dnbr_raster=idx["dnbr"])

        all_patches.extend(patches)
        scenes_data.append({
            "t1": t1, "t2": t2, "mask": mask, "dnbr": idx["dnbr"],
            "patches": patches, "labeled_mask": labeled_mask
        })

    print(f"  • Ingested {len(test_before)} test scenes")
    print(f"  • Extracted {len(all_patches):,} discrete disturbance patches across landscapes")

    # 3. Classify Disturbance Drivers
    print("\n[3/5] Classifying Spatial Disturbance Drivers (Roads, Fires, Logging, Clearcuts)...")
    df_patches = driver_clf.classify_all_patches(all_patches)
    driver_stats = driver_clf.compute_driver_statistics(df_patches)

    print("\n" + "=" * 90)
    print("📊 ANTHROPOGENIC & NATURAL DISTURBANCE DRIVER AUDIT")
    print("=" * 90)
    print(f"{'Disturbance Driver':<26} {'Patches':<10} {'Area (ha)':<14} {'Mean Patch (ha)':<18} {'Share (%)':<10}")
    print("-" * 90)
    for driver, s in driver_stats.items():
        print(f"{driver:<26} {s['patch_count']:<10,} {s['total_area_ha']:<14,} {s['mean_patch_ha']:<18} {s['area_percentage']:>5.2f}%")
    print("=" * 90)

    out_dir = "outputs/module_14"
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "disturbance_patches_manifest.csv")
    df_patches.to_csv(manifest_path, index=False)
    print(f"  💾 Saved patches manifest -> '{manifest_path}'")

    # 4. Forest Fragmentation & Edge Effect Analysis (Scene 1)
    print("\n[4/5] Computing 100m Forest Edge Buffer & Fragmentation Metrics (Scene 1)...")
    scene_1 = scenes_data[0]
    # Derive forest mask at T1: NDVI >= 0.50
    t1_ndvi = calc.compute_ndvi(scene_1["t1"][3], scene_1["t1"][2])
    forest_t1 = (t1_ndvi >= 0.50).astype(np.uint8)

    frag_metrics, frag_map = frag_analyzer.analyze_fragmentation(
        forest_mask=forest_t1,
        disturbance_mask=scene_1["mask"]
    )

    print("\n" + "=" * 80)
    print("🌳 FOREST INTACTNESS & EDGE EFFECT AUDIT (100m BUFFER)")
    print("=" * 80)
    print(f"  • Total Forest Remaining   : {frag_metrics['total_forest_remaining_ha']:,} ha")
    print(f"  • Core Interior Forest     : {frag_metrics['core_forest_ha']:,} ha ({frag_metrics['core_forest_percentage']:.1f}% intact)")
    print(f"  • Degraded Edge Forest     : {frag_metrics['edge_forest_ha']:,} ha ({frag_metrics['edge_forest_percentage']:.1f}% exposed)")
    print(f"  • Edge-to-Core Ratio       : {frag_metrics['edge_to_core_ratio']:.3f}")
    print(f"  • Disconnected Fragments   : {frag_metrics['number_of_forest_fragments']} forest patches")
    print("=" * 80)

    # Save JSON Summary
    summary_json = {
        "driver_statistics": driver_stats,
        "fragmentation_scene_1": frag_metrics
    }
    with open(os.path.join(out_dir, "pattern_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    # 5. Generate Visualizations
    print(f"\n[5/5] Generating Landscape Pattern Visualizations in '{out_dir}/'...")

    # --------------------------------------------------------------------------
    # Figure 1: Spatial Pattern Driver Map
    # --------------------------------------------------------------------------
    rgb_t1 = make_rgb(scene_1["t1"])
    rgb_t2 = make_rgb(scene_1["t2"])

    # Create driver colored overlay on T2
    overlay = rgb_t2.copy()
    h, w, _ = overlay.shape

    # Color each patch contour by driver
    driver_color_rgb = {
        "Road Encroachment": (1.0, 0.9, 0.1),     # Yellow
        "Wildfire Scar": (0.9, 0.15, 0.15),       # Red
        "Selective Logging": (0.0, 0.85, 0.95),   # Cyan
        "Agricultural Clearcut": (1.0, 0.45, 0.0) # Deep Orange
    }

    driver_map_canvas = np.zeros((h, w, 3), dtype=np.float32)

    for p in scene_1["patches"]:
        driver = driver_clf.classify_single_patch(p)
        c = driver_color_rgb[driver]
        cnt = p["contour"]
        # Draw on overlay and map canvas
        cv2.drawContours(driver_map_canvas, [cnt], -1, c, thickness=-1)

    # Blend overlay with RGB: 65% satellite image, 35% colored polygon
    blended = np.where(driver_map_canvas > 0, 0.55 * rgb_t2 + 0.45 * driver_map_canvas, rgb_t2)

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True)

    ax1.imshow(rgb_t2)
    ax1.set_title("Post-Disturbance (T2) Satellite Scene", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(blended)
    ax2.set_title("Spatial Disturbance Driver Classification Overlay", fontsize=11, fontweight="bold")
    ax2.axis("off")

    legend_patches = [
        mpatches.Patch(color=driver_color_rgb[d], label=f"{d} ({driver_stats[d]['area_percentage']:.1f}%)")
        for d in driver_clf.DRIVERS
    ]
    ax2.legend(handles=legend_patches, loc="lower right", framealpha=0.9, fontsize=9)

    # Driver Area Bar Chart
    drivers_list = driver_clf.DRIVERS
    areas_list = [driver_stats[d]["total_area_ha"] for d in drivers_list]
    colors_bar = [driver_clf.DRIVER_COLORS[d] for d in drivers_list]

    bars = ax3.bar(drivers_list, areas_list, color=colors_bar, edgecolor="black", alpha=0.9, width=0.55)
    ax3.set_ylabel("Total Disturbed Area (Hectares)", fontsize=11, fontweight="bold")
    ax3.set_title("Total Forest Loss by Disturbance Driver", fontsize=12, fontweight="bold")
    ax3.set_xticklabels(drivers_list, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h_val = bar.get_height()
        ax3.annotate(f"{h_val:.1f} ha", xy=(bar.get_x() + bar.get_width() / 2, h_val),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig1_path = os.path.join(out_dir, "01_spatial_pattern_driver_map.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  📸 Saved spatial pattern driver map -> '{fig1_path}'")

    # --------------------------------------------------------------------------
    # Figure 2: Morphological Feature Distributions
    # --------------------------------------------------------------------------
    fig2, (ax2_1, ax2_2, ax2_3) = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    for driver in driver_clf.DRIVERS:
        sub = df_patches[df_patches["driver"] == driver]
        if len(sub) == 0:
            continue
        c = driver_clf.DRIVER_COLORS[driver]

        # Scatter 1: Linearity vs Circularity
        ax2_1.scatter(sub["circularity"], sub["linearity"], color=c, label=driver, s=60, alpha=0.75, edgecolors="none")

        # Scatter 2: Area (ha) vs Fractal Dimension
        ax2_2.scatter(sub["area_hectares"], sub["fractal_dimension"], color=c, label=driver, s=60, alpha=0.75, edgecolors="none")

        # Scatter 3: Solidity vs Mean dNBR
        ax2_3.scatter(sub["solidity"], sub["mean_dnbr"], color=c, label=driver, s=60, alpha=0.75, edgecolors="none")

    ax2_1.set_xlabel("Circularity (Isoperimetric Quotient)", fontweight="bold")
    ax2_1.set_ylabel("Linearity / Aspect Ratio (L/W)", fontweight="bold")
    ax2_1.set_title("Linearity vs Circularity\n(Roads separate at high linearity)", fontweight="bold")
    ax2_1.grid(True, linestyle="--", alpha=0.5)
    ax2_1.legend(loc="upper right", fontsize=8)

    ax2_2.set_xlabel("Patch Area (Hectares)", fontweight="bold")
    ax2_2.set_ylabel("Fractal Dimension (Boundary Complexity)", fontweight="bold")
    ax2_2.set_title("Area vs Fractal Dimension\n(Wildfires show high fractal perimeters)", fontweight="bold")
    ax2_2.grid(True, linestyle="--", alpha=0.5)

    ax2_3.set_xlabel("Solidity (Convex Area Ratio)", fontweight="bold")
    ax2_3.set_ylabel("Mean dNBR (Burn Severity)", fontweight="bold")
    ax2_3.set_title("Solidity vs Burn Severity\n(Clearcuts show high solidity)", fontweight="bold")
    ax2_3.grid(True, linestyle="--", alpha=0.5)

    fig2_path = os.path.join(out_dir, "02_morphological_feature_distributions.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  📸 Saved morphological distributions -> '{fig2_path}'")

    # --------------------------------------------------------------------------
    # Figure 3: Forest Fragmentation & Edge Effect Map
    # --------------------------------------------------------------------------
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    # Colormap: 0=Non-forest/Deforested (Gray), 1=Edge Forest (Orange/Yellow), 2=Core Forest (Dark Green)
    cmap_frag = plt.matplotlib.colors.ListedColormap(["#424242", "#ffb300", "#1b5e20"])
    bounds_frag = [-0.5, 0.5, 1.5, 2.5]
    norm_frag = plt.matplotlib.colors.BoundaryNorm(bounds_frag, cmap_frag.N)

    im_frag = ax3_1.imshow(frag_map, cmap=cmap_frag, norm=norm_frag)
    ax3_1.set_title("Forest Fragmentation & 100m Edge Buffer Map", fontsize=12, fontweight="bold")
    ax3_1.axis("off")

    cbar_frag = plt.colorbar(im_frag, ax=ax3_1, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar_frag.ax.set_yticklabels(["Non-Forest / Cleared", "Edge Forest (<=100m)", "Core Forest (>100m)"], fontsize=9, fontweight="bold")

    # Intactness Breakdown Bar Chart
    cat_names = ["Core Forest (>100m)", "Edge Forest (<=100m)", "New Deforestation"]
    cat_areas = [frag_metrics["core_forest_ha"], frag_metrics["edge_forest_ha"], frag_metrics["new_deforestation_ha"]]
    cat_colors = ["#1b5e20", "#ffb300", "#d32f2f"]

    bars_f = ax3_2.bar(cat_names, cat_areas, color=cat_colors, edgecolor="black", alpha=0.85, width=0.5)
    ax3_2.set_ylabel("Landscape Footprint (Hectares)", fontsize=11, fontweight="bold")
    ax3_2.set_title("Forest Intactness & Edge Exposure Audit", fontsize=12, fontweight="bold")
    ax3_2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars_f:
        h_val = bar.get_height()
        ax3_2.annotate(f"{h_val:.1f} ha", xy=(bar.get_x() + bar.get_width() / 2, h_val),
                       xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig3_path = os.path.join(out_dir, "03_forest_fragmentation_edge_effect.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  📸 Saved forest fragmentation map -> '{fig3_path}'")

    print("\n" + "=" * 80)
    print("✅ MODULE 14: PATTERN ANALYSIS & FRAGMENTATION COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
