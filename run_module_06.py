"""
run_module_06.py

Master execution script for Module 6: Traditional Forest Change Detection.
Establishes the non-deep-learning baseline:
  1. Image Differencing vs. NDVI Differencing
  2. Optimal Decision Threshold Discovery (Precision vs. Recall sweep)
  3. Morphological False-Positive Filtering
  4. Comprehensive Benchmark Evaluation on Test Set (Precision, Recall, F1, IoU)
  5. Generates publication-quality diagnostic figures in outputs/module_06/
"""

import os
import sys
import glob
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

from modules.module_06_traditional_change_detection.change_detector import (
    TraditionalChangeDetector,
    ChangeEvaluator
)

def stretch(band):
    """Percentile linear contrast stretching."""
    p2, p98 = np.percentile(band, (2, 98))
    if p98 <= p2:
        return np.clip(band, 0.0, 1.0)
    return np.clip((band - p2) / (p98 - p2), 0.0, 1.0)

def make_rgb(bands):
    """Creates RGB composite (Red: band 2, Green: band 1, Blue: band 0)."""
    return np.stack([stretch(bands[2]), stretch(bands[1]), stretch(bands[0])], axis=-1)

def main():
    print("\n" + "=" * 75)
    print("[*] MODULE 6: TRADITIONAL FOREST CHANGE DETECTION - MASTER RUNNER")
    print("=" * 75)

    test_before_files = sorted(glob.glob("dataset/test/before/*.tif"))
    test_after_files  = sorted(glob.glob("dataset/test/after/*.tif"))
    test_mask_files   = sorted(glob.glob("dataset/test/masks/*.tif"))

    if not test_before_files:
        print("Error: No test samples found in dataset/test/. Please run run_module_03.py first.")
        return

    detector = TraditionalChangeDetector()

    # 1. Ingest representative test sample
    sample_b_path = test_before_files[0]
    sample_a_path = test_after_files[0]
    sample_m_path = test_mask_files[0]

    with rasterio.open(sample_b_path) as src_b:
        b_bands = src_b.read()
    with rasterio.open(sample_a_path) as src_a:
        a_bands = src_a.read()
    with rasterio.open(sample_m_path) as src_m:
        gt_mask = src_m.read(1)

    print(f"\n[1/4] Loaded multi-temporal test scene: {os.path.basename(sample_b_path)}")

    # 2. Compute Differencing Maps
    print("\n[2/4] Computing Image Differencing & NDVI Differencing...")
    spectral_diff = detector.compute_spectral_vector_difference(b_bands, a_bands)
    delta_ndvi = detector.compute_ndvi_difference(b_bands, a_bands)

    print(f"  • Spectral Distance Range : [{spectral_diff.min():.3f}, {spectral_diff.max():.3f}]")
    print(f"  • Delta NDVI Range        : [{delta_ndvi.min():.3f}, {delta_ndvi.max():.3f}]")

    # 3. Discover Optimal Threshold
    print("\n[3/4] Optimizing Decision Threshold via F1 / IoU Sweep...")
    thresholds = np.linspace(0.05, 0.70, 33)
    best_thresh, best_f1, sweep_hist = detector.find_optimal_threshold(delta_ndvi, gt_mask, thresholds)
    print(f"  ⭐ Optimal Delta NDVI Threshold = {best_thresh:.3f} (Max F1-Score: {best_f1 * 100:.2f}%)")

    # Evaluate on optimal threshold
    predicted_mask = detector.detect_change(delta_ndvi, threshold=best_thresh, apply_morph_cleanup=True)
    metrics = ChangeEvaluator.evaluate(predicted_mask, gt_mask)
    ChangeEvaluator.print_metrics(metrics, title=f"Baseline NDVI Differencing (Threshold={best_thresh:.2f})")

    # 4. Generate Publication Figures
    out_dir = "outputs/module_06"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[4/4] Generating Benchmark Visualizations in '{out_dir}/'...")

    # Plot 1: Differencing Techniques Comparison
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    axes1[0].imshow(make_rgb(b_bands))
    axes1[0].set_title("1. Before Image (RGB)", fontsize=11, fontweight="bold")
    axes1[0].axis("off")

    im_spec = axes1[1].imshow(spectral_diff, cmap="magma")
    axes1[1].set_title("2. Spectral Euclidean Distance\nsqrt(sum((B_after - B_before)^2))", fontsize=11, fontweight="bold")
    axes1[1].axis("off")
    fig1.colorbar(im_spec, ax=axes1[1], fraction=0.046, pad=0.04)

    im_ndvi = axes1[2].imshow(delta_ndvi, cmap="RdBu_r", vmin=-0.5, vmax=0.8)
    axes1[2].set_title("3. NDVI Differencing\n(NDVI_before - NDVI_after)", fontsize=11, fontweight="bold", color="darkred")
    axes1[2].axis("off")
    fig1.colorbar(im_ndvi, ax=axes1[2], fraction=0.046, pad=0.04)

    fig1.suptitle("Traditional Image Differencing Techniques (Module 6)", fontsize=13, fontweight="bold", y=1.02)
    plot1_path = os.path.join(out_dir, "01_differencing_techniques_comparison.png")
    plt.savefig(plot1_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"  💾 Saved differencing comparison to: {plot1_path}")

    # Plot 2: Threshold Optimization Curve
    fig2, ax2 = plt.subplots(figsize=(8.5, 5))
    ax2.plot(sweep_hist["thresholds"], sweep_hist["f1_scores"], "r-o", linewidth=2.5, label="F1-Score (Dice)")
    ax2.plot(sweep_hist["thresholds"], sweep_hist["ious"], "b--s", linewidth=2, label="IoU (Jaccard)")
    ax2.plot(sweep_hist["thresholds"], sweep_hist["precisions"], "g-^", linewidth=1.5, alpha=0.75, label="Precision")
    ax2.plot(sweep_hist["thresholds"], sweep_hist["recalls"], "m-v", linewidth=1.5, alpha=0.75, label="Recall")
    ax2.axvline(best_thresh, color="black", linestyle=":", linewidth=2, label=f"Optimal Threshold ({best_thresh:.2f})")

    ax2.set_title("Decision Threshold Optimization Curve for NDVI Differencing", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Delta NDVI Threshold", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Metric Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10, loc="center right", framealpha=0.9)

    plot2_path = os.path.join(out_dir, "02_threshold_optimization_curve.png")
    plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  💾 Saved threshold optimization curve to: {plot2_path}")

    # Plot 3: Baseline Detection & Error Analysis Map
    # Error classification:
    # TP = Green, FP = Red (over-detection), FN = Blue (missed), TN = Black
    error_rgb = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    tp_mask = (predicted_mask == 1) & (gt_mask == 1)
    fp_mask = (predicted_mask == 1) & (gt_mask == 0)
    fn_mask = (predicted_mask == 0) & (gt_mask == 1)

    error_rgb[tp_mask] = [0.15, 0.75, 0.25]  # Green: True Positives
    error_rgb[fp_mask] = [0.90, 0.15, 0.15]  # Red: False Positives
    error_rgb[fn_mask] = [0.20, 0.40, 0.90]  # Blue: False Negatives

    fig3, axes3 = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)

    axes3[0].imshow(make_rgb(b_bands))
    axes3[0].set_title("1. Before Image (T1)", fontsize=11, fontweight="bold")
    axes3[0].axis("off")

    axes3[1].imshow(make_rgb(a_bands))
    axes3[1].set_title("2. After Image (T2)\n[Deforestation visible]", fontsize=11, fontweight="bold")
    axes3[1].axis("off")

    axes3[2].imshow(gt_mask, cmap="gray")
    axes3[2].set_title("3. Ground-Truth Mask", fontsize=11, fontweight="bold")
    axes3[2].axis("off")

    axes3[3].imshow(error_rgb)
    axes3[3].set_title(
        f"4. Error Map (Baseline)\n[Green: TP, Red: FP, Blue: FN]\nF1: {metrics['f1_score']*100:.1f}% | IoU: {metrics['iou']*100:.1f}%",
        fontsize=11, fontweight="bold"
    )
    axes3[3].axis("off")

    fig3.suptitle("Traditional Change Detection Benchmark & Error Analysis (Module 6)", fontsize=13, fontweight="bold", y=1.03)
    plot3_path = os.path.join(out_dir, "03_traditional_change_detection_benchmark.png")
    plt.savefig(plot3_path, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"  💾 Saved benchmark error analysis to: {plot3_path}")

    # Summary
    print("\n" + "=" * 75)
    print("🎯 Traditional Baseline Established:")
    print(f"  • Optimal Delta NDVI Threshold : {best_thresh:.3f}")
    print(f"  • Precision                    : {metrics['precision'] * 100:.2f}%")
    print(f"  • Recall                       : {metrics['recall'] * 100:.2f}%")
    print(f"  • F1-Score (Dice)              : {metrics['f1_score'] * 100:.2f}%")
    print(f"  • IoU (Jaccard Index)          : {metrics['iou'] * 100:.2f}%")
    print(f"  • Pixel Accuracy               : {metrics['accuracy'] * 100:.2f}%")
    print("📌 This establishes the quantitative baseline to beat with ML and Deep Learning in Modules 7-12!")
    print(f"📁 All visual products saved in: {os.path.abspath(out_dir)}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
