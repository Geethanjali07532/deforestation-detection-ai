"""
dataset_visualizer.py

Module 3: Dataset Collection & Organization
Visualizes multi-temporal satellite observation pairs:
  - Before Image (True Color & False Color CIR)
  - After Image (True Color & False Color CIR)
  - Ground-Truth Deforestation Mask
  - Direct Overlay of Detected Changes on the Satellite Image
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def stretch_band(band, p_min=2.0, p_max=98.0):
    """Percentile linear contrast stretching to [0, 1]."""
    v_min, v_max = np.percentile(band, (p_min, p_max))
    if v_max <= v_min:
        return np.clip(band, 0.0, 1.0)
    return np.clip((band - v_min) / (v_max - v_min), 0.0, 1.0)

def make_rgb(bands):
    """Creates RGB composite from 5-band array (1:B, 2:G, 3:R, 4:NIR, 5:SWIR)."""
    r = stretch_band(bands[2])  # Red
    g = stretch_band(bands[1])  # Green
    b = stretch_band(bands[0])  # Blue
    return np.stack([r, g, b], axis=-1)

def make_cir(bands):
    """Creates False Color CIR composite (NIR, Red, Green)."""
    r = stretch_band(bands[3])  # NIR
    g = stretch_band(bands[2])  # Red
    b = stretch_band(bands[1])  # Green
    return np.stack([r, g, b], axis=-1)

def visualize_temporal_triplet(
    before_path: str,
    after_path: str,
    mask_path: str,
    sample_title: str = "Multi-Temporal Deforestation Triplet",
    output_path: str = "outputs/module_03/02_multitemporal_pair_inspection.png"
):
    """Plots and saves side-by-side comparison of Before, After, Mask, and Overlay."""
    with rasterio.open(before_path) as src_b:
        b_data = src_b.read()
    with rasterio.open(after_path) as src_a:
        a_data = src_a.read()
    with rasterio.open(mask_path) as src_m:
        mask_data = src_m.read(1)

    b_rgb = make_rgb(b_data)
    a_rgb = make_rgb(a_data)
    b_cir = make_cir(b_data)
    a_cir = make_cir(a_data)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    # Row 1: True Color & Masks
    # 1. Before True Color
    axes[0, 0].imshow(b_rgb)
    axes[0, 0].set_title("1. Before (True Color RGB)\n[Intact Forest Canopy]", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. After True Color
    axes[0, 1].imshow(a_rgb)
    axes[0, 1].set_title("2. After (True Color RGB)\n[New Clearings & Road Encroachment]", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # 3. Ground Truth Mask
    im_mask = axes[0, 2].imshow(mask_data, cmap="viridis", vmin=0, vmax=1)
    deforest_pct = (np.sum(mask_data == 1) / mask_data.size) * 100
    axes[0, 2].set_title(f"3. Ground-Truth Deforestation Mask\n[1 = Deforested ({deforest_pct:.1f}%), 0 = Unchanged]", fontsize=11, fontweight="bold", color="darkred")
    axes[0, 2].axis("off")

    # Row 2: False Color CIR & Change Overlay
    # 4. Before CIR
    axes[1, 0].imshow(b_cir)
    axes[1, 0].set_title("4. Before (False Color CIR)\n[Healthy Vegetation = Solid Red]", fontsize=11, fontweight="bold", color="crimson")
    axes[1, 0].axis("off")

    # 5. After CIR
    axes[1, 1].imshow(a_cir)
    axes[1, 1].set_title("5. After (False Color CIR)\n[Deforestation appears as Cyan/Grey]", fontsize=11, fontweight="bold", color="crimson")
    axes[1, 1].axis("off")

    # 6. Change Overlay on After Image
    axes[1, 2].imshow(a_rgb)
    # Highlight deforested pixels in bright red with semi-transparency
    overlay = np.zeros((*mask_data.shape, 4), dtype=np.float32)
    overlay[mask_data == 1] = [1.0, 0.0, 0.0, 0.65]  # Semi-transparent red
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("6. Deforestation Overlay on After Image\n[Red Highlight = Lost Forest Area]", fontsize=11, fontweight="bold", color="red")
    axes[1, 2].axis("off")

    fig.suptitle(f"{sample_title} (Module 3)", fontsize=14, fontweight="bold", y=1.03)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"💾 Saved multi-temporal visual inspection to: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    sample_b = "dataset/train/before/scene_train_001.tif"
    sample_a = "dataset/train/after/scene_train_001.tif"
    sample_m = "dataset/train/masks/scene_train_001.tif"

    if os.path.exists(sample_b):
        visualize_temporal_triplet(sample_b, sample_a, sample_m)
    else:
        print("Please build the dataset first using dataset_builder.py")
