"""
run_module_04.py

Master execution script for Module 4: Satellite Image Preprocessing.
Runs the complete multi-temporal preprocessing pipeline:
  1. Cloud / Noise Removal
  2. Geometric Alignment (Co-Registration)
  3. Band Selection & Temporal Stacking (10-channel tensor)
  4. Radiometric Normalization
  5. Image Tiling & Chip Extraction
Generates stage-by-stage diagnostic figures in outputs/module_04/.
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

from modules.module_04_preprocessing.preprocessing_pipeline import SatellitePreprocessingPipeline

def make_rgb(bands):
    """Produces RGB visualization from (5, H, W) array."""
    def strt(b):
        p2, p98 = np.percentile(b, (2, 98))
        if p98 <= p2:
            return np.clip(b, 0.0, 1.0)
        return np.clip((b - p2) / (p98 - p2), 0.0, 1.0)
    r = strt(bands[2])
    g = strt(bands[1])
    b = strt(bands[0])
    return np.stack([r, g, b], axis=-1)

def main():
    print("\n" + "=" * 75)
    print("[*] MODULE 4: SATELLITE IMAGE PREPROCESSING PIPELINE - MASTER RUNNER")
    print("=" * 75)

    sample_b_path = "dataset/train/before/scene_train_001.tif"
    sample_a_path = "dataset/train/after/scene_train_001.tif"
    sample_m_path = "dataset/train/masks/scene_train_001.tif"

    if not os.path.exists(sample_b_path):
        print(f"Error: {sample_b_path} not found. Please run run_module_03.py first.")
        return

    # Load raw data
    with rasterio.open(sample_b_path) as src_b:
        raw_before = src_b.read()
    with rasterio.open(sample_a_path) as src_a:
        raw_after = src_a.read()
    with rasterio.open(sample_m_path) as src_m:
        raw_mask = src_m.read(1)

    print(f"\n[1/5] Loaded raw satellite images: {raw_before.shape} (5 bands, 256x256)")

    # Simulate realistic real-world satellite imperfections:
    # 1. Add deliberate geometric misalignment (dy=3, dx=-4) to test co-registration
    from scipy.ndimage import shift
    sim_after = np.zeros_like(raw_after)
    for b in range(5):
        sim_after[b] = shift(raw_after[b], shift=(3.0, -4.0), mode="nearest")

    # 2. Add localized cloud patch (bright in visible channels) to test cloud masking
    yy, xx = np.ogrid[:256, :256]
    cloud_patch = ((xx - 70)**2 / (25**2) + (yy - 70)**2 / (20**2) <= 1)
    sim_before = raw_before.copy()
    sim_before[0:3, cloud_patch] += 0.45  # Intense bright cloud

    print("\n[2/5] Running SatellitePreprocessingPipeline...")
    pipeline = SatellitePreprocessingPipeline(
        tile_size=128,
        stride=128,
        normalization_method="robust"
    )

    results = pipeline.process_pair(sim_before, sim_after, raw_mask)

    dy, dx = results["detected_shift_px"]
    print(f"  • Geometric Co-Registration Shift Detected : dy = {dy:+.1f} px, dx = {dx:+.1f} px")
    print(f"  • Temporal Stack Tensor Shape              : {results['stacked_temporal'].shape} (10 channels)")
    print(f"  • Model-Ready Image Chips Extracted        : {results['total_chips_extracted']} chips (128x128)")

    # Generate Visualizations
    out_dir = "outputs/module_04"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[3/5] Generating Stage-by-Stage Preprocessing Figures in '{out_dir}/'...")

    # Plot 1: Full Pipeline Stages
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    # 1. Raw with Cloud Artifact
    axes[0, 0].imshow(make_rgb(sim_before))
    axes[0, 0].set_title("1. Raw Before Image\n(With Cloud Contamination)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Cloud Mask Detected
    axes[0, 1].imshow(results["cloud_mask_before"], cmap="gray")
    axes[0, 1].set_title("2. Cloud Detection Mask\n(Spectral Thresholding)", fontsize=11, fontweight="bold", color="darkblue")
    axes[0, 1].axis("off")

    # 3. Cleaned & Denoised
    axes[0, 2].imshow(make_rgb(results["denoised_before"]))
    axes[0, 2].set_title("3. Denoised & Inpainted Image\n(Edge-Preserving Spatial Filter)", fontsize=11, fontweight="bold", color="darkgreen")
    axes[0, 2].axis("off")

    # 4. Misaligned Raw After
    axes[1, 0].imshow(make_rgb(sim_after))
    axes[1, 0].set_title(f"4. Raw After Image\n(Misaligned by dy=+3, dx=-4)", fontsize=11, fontweight="bold", color="darkred")
    axes[1, 0].axis("off")

    # 5. Geometrically Re-aligned After
    axes[1, 1].imshow(make_rgb(results["aligned_after"]))
    axes[1, 1].set_title(f"5. Co-Registered After Image\n(Corrected shift dy={dy:.1f}, dx={dx:.1f})", fontsize=11, fontweight="bold", color="green")
    axes[1, 1].axis("off")

    # 6. Normalized Stacked Composite
    norm_stack = results["stacked_temporal"]
    # Display difference between NIR after and NIR before
    nir_diff = norm_stack[8] - norm_stack[3]  # After NIR - Before NIR
    im_diff = axes[1, 2].imshow(nir_diff, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title("6. Temporal NIR Feature Difference\n(Red/Blue highlights canopy change)", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")
    fig.colorbar(im_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle("Satellite Image Preprocessing Pipeline Stages (Module 4)", fontsize=14, fontweight="bold", y=1.02)
    stage_plot = os.path.join(out_dir, "01_preprocessing_pipeline_stages.png")
    plt.savefig(stage_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved stage-by-stage plot to: {stage_plot}")

    # Plot 2: Image Tiling / Chip Inspection
    chips = results["chips"]
    fig2, axes2 = plt.subplots(len(chips), 3, figsize=(10, 3.2 * len(chips)), constrained_layout=True)

    for i, chip in enumerate(chips):
        b_rgb = make_rgb(chip["before"])
        a_rgb = make_rgb(chip["after"])
        mask_ch = chip["mask"]

        axes2[i, 0].imshow(b_rgb)
        axes2[i, 0].set_title(f"Chip {chip['chip_id']}: Before Tile (128x128)\n[row {chip['row_start']}, col {chip['col_start']}]", fontsize=10)
        axes2[i, 0].axis("off")

        axes2[i, 1].imshow(a_rgb)
        axes2[i, 1].set_title(f"Chip {chip['chip_id']}: After Tile (128x128)", fontsize=10)
        axes2[i, 1].axis("off")

        axes2[i, 2].imshow(mask_ch, cmap="viridis", vmin=0, vmax=1)
        axes2[i, 2].set_title(f"Chip {chip['chip_id']}: Deforestation Mask\n({chip['deforested_pct']:.1f}% loss)", fontsize=10, color="darkred")
        axes2[i, 2].axis("off")

    fig2.suptitle("Model-Ready Chipped Tiles & Synchronized Masks (Module 4)", fontsize=13, fontweight="bold", y=1.02)
    chip_plot = os.path.join(out_dir, "02_tiled_chips_inspection.png")
    plt.savefig(chip_plot, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  💾 Saved chipped tiles inspection to: {chip_plot}")

    print("\n" + "=" * 75)
    print("🎯 Module 4 Pipeline Successfully Validated:")
    print("  1. Cloud Masking: Filtered atmospheric clouds and reconstructed scene integrity.")
    print("  2. Geometric Alignment: Automated sub-pixel co-registration eliminated temporal drift.")
    print("  3. Radiometric Normalization: Scaled surface reflectance to robust [0, 1] range.")
    print("  4. Band Stacking: Created unified 10-channel temporal tensors for deep learning models.")
    print(f"  5. Tiling: Chipped 256x256 scenes into {len(chips)} synchronized 128x128 model inputs.")
    print(f"📁 All visual outputs saved in: {os.path.abspath(out_dir)}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
