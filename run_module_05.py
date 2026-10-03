"""
run_module_05.py

Master execution script for Module 5: Vegetation Index Analysis.
Executes:
  1. Calculation of spectral indices: NDVI, EVI, SAVI, NDWI (Moisture & Water), NBR
  2. Generation of the 4-class discrete Vegetation & Land Cover Map
  3. Binary Forest vs. Non-Forest segmentation
  4. Temporal index differencing (Delta NDVI preview)
  5. Publication-quality figures in outputs/module_05/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio

# Ensure UTF-8 output encoding for Windows command line / PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure local module directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.module_05_vegetation_indices.vegetation_indices import (
    VegetationIndexCalculator,
    VegetationClassifier
)

def stretch_band(band, p_min=2.0, p_max=98.0):
    """Percentile linear contrast stretching."""
    v_min, v_max = np.percentile(band, (p_min, p_max))
    if v_max <= v_min:
        return np.clip(band, 0.0, 1.0)
    return np.clip((band - v_min) / (v_max - v_min), 0.0, 1.0)

def make_rgb(bands):
    """Creates RGB composite (Red: band 2, Green: band 1, Blue: band 0)."""
    return np.stack([stretch_band(bands[2]), stretch_band(bands[1]), stretch_band(bands[0])], axis=-1)

def main():
    print("\n" + "=" * 75)
    print("[*] MODULE 5: VEGETATION INDEX ANALYSIS - MASTER RUNNER")
    print("=" * 75)

    sample_b_path = "dataset/train/before/scene_train_001.tif"
    sample_a_path = "dataset/train/after/scene_train_001.tif"
    sample_m_path = "dataset/train/masks/scene_train_001.tif"

    if not os.path.exists(sample_b_path):
        print(f"Error: {sample_b_path} not found. Please run run_module_03.py first.")
        return

    with rasterio.open(sample_b_path) as src_b:
        before_bands = src_b.read()
    with rasterio.open(sample_a_path) as src_a:
        after_bands = src_a.read()
    with rasterio.open(sample_m_path) as src_m:
        ground_truth_mask = src_m.read(1)

    print(f"\n[1/4] Loaded multi-spectral satellite scenes: {before_bands.shape}")

    # 1. Compute Vegetation Indices
    print("\n[2/4] Computing Remote Sensing Indices (NDVI, EVI, SAVI, NDWI, NBR)...")
    calc_b = VegetationIndexCalculator(before_bands)
    calc_a = VegetationIndexCalculator(after_bands)

    indices_b = calc_b.compute_all_indices()
    indices_a = calc_a.compute_all_indices()

    ndvi_b = indices_b["NDVI"]
    ndvi_a = indices_a["NDVI"]
    evi_b = indices_b["EVI"]
    savi_b = indices_b["SAVI"]
    ndwi_m_b = indices_b["NDWI_Moisture"]
    nbr_b = indices_b["NBR"]

    print("  • NDVI Range : [{:.3f}, {:.3f}], Mean = {:.3f}".format(ndvi_b.min(), ndvi_b.max(), ndvi_b.mean()))
    print("  • EVI Range  : [{:.3f}, {:.3f}], Mean = {:.3f}".format(evi_b.min(), evi_b.max(), evi_b.mean()))
    print("  • SAVI Range : [{:.3f}, {:.3f}], Mean = {:.3f}".format(savi_b.min(), savi_b.max(), savi_b.mean()))
    print("  • NDWI Range : [{:.3f}, {:.3f}], Mean = {:.3f}".format(ndwi_m_b.min(), ndwi_m_b.max(), ndwi_m_b.mean()))
    print("  • NBR Range  : [{:.3f}, {:.3f}], Mean = {:.3f}".format(nbr_b.min(), nbr_b.max(), nbr_b.mean()))

    # 2. Vegetation Classification & Land Cover Mapping
    print("\n[3/4] Generating Discrete Vegetation & Land Cover Map...")
    classifier = VegetationClassifier(forest_threshold=0.50)
    veg_map_b = classifier.create_vegetation_map(ndvi_b)
    binary_forest_b = classifier.binary_forest_mask(ndvi_b)

    stats = classifier.compute_class_statistics(veg_map_b)
    print("\n" + "-" * 75)
    print(f"{'LAND COVER CLASS':<28} | {'PIXELS':<8} | {'PERCENT':<8} | {'AREA (HA)':<10}")
    print("-" * 75)
    for class_name, s in stats.items():
        print(f"{class_name:<28} | {s['pixel_count']:<8} | {s['percentage']:>6.2f}% | {s['area_ha']:>8.2f} ha")
    print("-" * 75)

    # 3. Generate Visualizations
    out_dir = "outputs/module_05"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[4/4] Generating Publication Figures in '{out_dir}/'...")

    # Plot 1: All 5 Vegetation Indices Comparison
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    axes1[0, 0].imshow(make_rgb(before_bands))
    axes1[0, 0].set_title("1. True Color RGB Reference", fontsize=11, fontweight="bold")
    axes1[0, 0].axis("off")

    im1 = axes1[0, 1].imshow(ndvi_b, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    axes1[0, 1].set_title("2. NDVI (Normalized Difference Vegetation Index)\n(NIR - Red) / (NIR + Red)", fontsize=11, fontweight="bold")
    axes1[0, 1].axis("off")
    fig1.colorbar(im1, ax=axes1[0, 1], fraction=0.046, pad=0.04)

    im2 = axes1[0, 2].imshow(evi_b, cmap="YlGn", vmin=0.0, vmax=1.0)
    axes1[0, 2].set_title("3. EVI (Enhanced Vegetation Index)\nCorrects for aerosol scattering & saturation", fontsize=11, fontweight="bold")
    axes1[0, 2].axis("off")
    fig1.colorbar(im2, ax=axes1[0, 2], fraction=0.046, pad=0.04)

    im3 = axes1[1, 0].imshow(savi_b, cmap="YlGn", vmin=0.0, vmax=0.9)
    axes1[1, 0].set_title("4. SAVI (Soil Adjusted Vegetation Index)\nSuppresses bright soil background noise", fontsize=11, fontweight="bold")
    axes1[1, 0].axis("off")
    fig1.colorbar(im3, ax=axes1[1, 0], fraction=0.046, pad=0.04)

    im4 = axes1[1, 1].imshow(ndwi_m_b, cmap="BrBG", vmin=-0.6, vmax=0.8)
    axes1[1, 1].set_title("5. NDWI Moisture (Canopy Water Content)\n(NIR - SWIR) / (NIR + SWIR)", fontsize=11, fontweight="bold")
    axes1[1, 1].axis("off")
    fig1.colorbar(im4, ax=axes1[1, 1], fraction=0.046, pad=0.04)

    im5 = axes1[1, 2].imshow(nbr_b, cmap="copper", vmin=-0.4, vmax=0.8)
    axes1[1, 2].set_title("6. NBR (Normalized Burn Ratio)\nSensitive to charcoal, fires, and canopy loss", fontsize=11, fontweight="bold")
    axes1[1, 2].axis("off")
    fig1.colorbar(im5, ax=axes1[1, 2], fraction=0.046, pad=0.04)

    fig1.suptitle("Remote Sensing Spectral Indices Suite (Module 5)", fontsize=14, fontweight="bold", y=1.02)
    plot1_path = os.path.join(out_dir, "01_all_vegetation_indices_comparison.png")
    plt.savefig(plot1_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"  💾 Saved indices comparison to: {plot1_path}")

    # Plot 2: Practical Roadmap Workflow: Satellite Image -> NDVI -> Vegetation Map
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    # 1. Satellite Image
    axes2[0].imshow(make_rgb(before_bands))
    axes2[0].set_title("1. Satellite Image (RGB)", fontsize=12, fontweight="bold")
    axes2[0].axis("off")

    # 2. NDVI Calculation
    im_ndvi = axes2[1].imshow(ndvi_b, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    axes2[1].set_title("2. NDVI Calculation (Continuous)\n[Range: -1.0 to +1.0]", fontsize=12, fontweight="bold")
    axes2[1].axis("off")
    cbar = fig2.colorbar(im_ndvi, ax=axes2[1], fraction=0.046, pad=0.04)
    cbar.set_label("NDVI Value", fontsize=10)

    # 3. Discrete Vegetation Map
    from matplotlib.colors import ListedColormap
    cmap_custom = ListedColormap(["#2980b9", "#d35400", "#f1c40f", "#27ae60"])
    im_map = axes2[2].imshow(veg_map_b, cmap=cmap_custom, vmin=0, vmax=3)
    axes2[2].set_title("3. Classified Vegetation Map\n[Discrete Land-Cover Classes]", fontsize=12, fontweight="bold", color="darkgreen")
    axes2[2].axis("off")

    cbar2 = fig2.colorbar(im_map, ax=axes2[2], fraction=0.046, pad=0.04, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar2.ax.set_yticklabels(["Water", "Soil/Cleared", "Degraded/Sparse", "Dense Forest"], fontsize=9, fontweight="bold")

    fig2.suptitle("Practical Pipeline: Satellite Image -> NDVI Calculation -> Vegetation Map", fontsize=13, fontweight="bold", y=1.02)
    plot2_path = os.path.join(out_dir, "02_ndvi_vegetation_classification_map.png")
    plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  💾 Saved practical vegetation map to: {plot2_path}")

    # Plot 3: Temporal Vegetation Change Analysis (Delta NDVI)
    delta_ndvi = ndvi_a - ndvi_b
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    axes3[0].imshow(ndvi_b, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    axes3[0].set_title("NDVI Before (T1)", fontsize=11, fontweight="bold")
    axes3[0].axis("off")

    axes3[1].imshow(ndvi_a, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    axes3[1].set_title("NDVI After (T2)", fontsize=11, fontweight="bold")
    axes3[1].axis("off")

    # Delta NDVI: Negative values indicate forest clearing!
    im_d = axes3[2].imshow(delta_ndvi, cmap="RdBu", vmin=-0.8, vmax=0.8)
    axes3[2].set_title("Temporal Delta NDVI (T2 - T1)\n[Deep Red = Massive Forest Loss]", fontsize=11, fontweight="bold", color="darkred")
    axes3[2].axis("off")
    cbar_d = fig3.colorbar(im_d, ax=axes3[2], fraction=0.046, pad=0.04)
    cbar_d.set_label("NDVI Difference", fontsize=10)

    fig3.suptitle("Temporal Deforestation Detection via NDVI Differencing (Module 5)", fontsize=13, fontweight="bold", y=1.02)
    plot3_path = os.path.join(out_dir, "03_temporal_vegetation_change_analysis.png")
    plt.savefig(plot3_path, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"  💾 Saved temporal change analysis to: {plot3_path}")

    print("\n" + "=" * 75)
    print("🎯 Module 5 Execution Complete!")
    print("  1. Computed all 5 remote-sensing indices: NDVI, EVI, SAVI, NDWI, NBR.")
    print("  2. Built the exact practical workflow: Satellite Image -> NDVI -> Vegetation Map.")
    print("  3. Quantified land cover distributions and forest canopy area in hectares.")
    print(f"📁 All visual products saved in: {os.path.abspath(out_dir)}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
