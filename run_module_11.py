"""
run_module_11.py

Master execution script for Module 11: Multi-Temporal Change Detection Using Siamese Neural Networks.
Accomplishes:
  1. Bi-temporal multispectral dataset loading (T1 Before, T2 After, Ground-Truth Mask)
  2. SiameseUNetChangeDetector initialization (Shared-weight encoder + multi-scale feature differencing)
  3. Optimization via CompoundChangeLoss (Weighted BCE + Dice Loss)
  4. Benchmark comparison: Siamese Deep Learning vs. Traditional Delta-NDVI thresholding
  5. High-resolution multi-scale feature differencing visualization
  6. Publication-grade figures in outputs/module_11/
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Ensure UTF-8 output encoding for Windows command line / PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure local module directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.module_11_siamese_change_detection.siamese_dataset import create_siamese_dataloaders
from modules.module_11_siamese_change_detection.siamese_model import SiameseUNetChangeDetector
from modules.module_11_siamese_change_detection.trainer import train_siamese_detector, evaluate_siamese_detector


def make_rgb(tensor_5ch):
    """Creates normalized RGB composite from (5, H, W) tensor."""
    arr = tensor_5ch.cpu().numpy()
    def strt(b):
        p2, p98 = np.percentile(b, (2, 98))
        if p98 <= p2:
            return np.clip(b, 0.0, 1.0)
        return np.clip((b - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
    # Red: band 2, Green: band 1, Blue: band 0
    return np.stack([strt(arr[2]), strt(arr[1]), strt(arr[0])], axis=-1)


def compute_traditional_delta_ndvi_metrics(test_loader, threshold: float = 0.50):
    """Evaluates traditional differencing on the same test set for fair comparison."""
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    t0 = time.time()

    for t1_b, t2_b, mask_b in test_loader:
        # NIR: band 3, Red: band 2
        nir1, red1 = t1_b[:, 3], t1_b[:, 2]
        nir2, red2 = t2_b[:, 3], t2_b[:, 2]

        ndvi1 = (nir1 - red1) / (nir1 + red1 + 1e-7)
        ndvi2 = (nir2 - red2) / (nir2 + red2 + 1e-7)
        delta_ndvi = ndvi1 - ndvi2

        preds_bin = (delta_ndvi >= threshold).numpy().astype(int).flatten()
        gt = mask_b.numpy().astype(int).flatten()

        total_tp += int(np.sum((preds_bin == 1) & (gt == 1)))
        total_fp += int(np.sum((preds_bin == 1) & (gt == 0)))
        total_tn += int(np.sum((preds_bin == 0) & (gt == 0)))
        total_fn += int(np.sum((preds_bin == 0) & (gt == 1)))

    total_time_ms = (time.time() - t0) * 1000.0
    latency_per_tile = total_time_ms / max(1, len(test_loader.dataset))

    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    dice = (2 * precision * recall) / (precision + recall + 1e-8)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "latency_ms": float(latency_per_tile)
    }


def main():
    print("\n" + "=" * 80)
    print("🛰️  MODULE 11: MULTI-TEMPORAL CHANGE DETECTION (SIAMESE NETWORKS)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/5] Compute device: [{device}]")

    # 1. Prepare Siamese DataLoaders
    print("\n[2/5] Preparing Bi-Temporal Paired DataLoaders (128x128 chips)...")
    train_loader, val_loader, test_loader = create_siamese_dataloaders(
        dataset_root="dataset",
        tile_size=128,
        batch_size=8,
        max_train_tiles=32,
        max_val_tiles=8
    )
    print(f"  • Train pairs : {len(train_loader.dataset)} bi-temporal pairs ({len(train_loader)} batches)")
    print(f"  • Val pairs   : {len(val_loader.dataset)} bi-temporal pairs ({len(val_loader)} batches)")
    print(f"  • Test pairs  : {len(test_loader.dataset)} bi-temporal pairs ({len(test_loader)} batches)")

    # 2. Instantiate Siamese Architecture
    print("\n[3/5] Instantiating SiameseUNetChangeDetector...")
    model = SiameseUNetChangeDetector(in_channels=5, out_channels=1, base_features=16)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • Trainable Parameters: {n_params:,}")

    # 3. Train Siamese Model
    print("\n[4/5] Training Siamese Network (Compound BCE + Dice Loss)...")
    trained_model, history = train_siamese_detector(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        lr=1e-3,
        device=device
    )

    # 4. Benchmark Evaluation
    print("[5/5] Evaluating Change Detection on Unseen Test Imagery...")
    siamese_metrics = evaluate_siamese_detector(trained_model, test_loader, device=device)
    trad_metrics = compute_traditional_delta_ndvi_metrics(test_loader, threshold=0.50)

    print("\n" + "=" * 85)
    print("📊 CHANGE DETECTION BENCHMARK: SIAMESE DEEP LEARNING VS TRADITIONAL DELTA-NDVI")
    print("=" * 85)
    print(f"{'Metric':<25} {'Traditional Delta-NDVI':<28} {'Siamese U-Net (Ours)':<25}")
    print("-" * 85)
    print(f"{'Mean IoU (Jaccard)':<25} {trad_metrics['iou']*100:>8.2f}%{' ':>20} {siamese_metrics['iou']*100:>8.2f}%")
    print(f"{'Dice Score (F1)':<25} {trad_metrics['dice']*100:>8.2f}%{' ':>20} {siamese_metrics['dice']*100:>8.2f}%")
    print(f"{'Pixel Accuracy':<25} {trad_metrics['accuracy']*100:>8.2f}%{' ':>20} {siamese_metrics['accuracy']*100:>8.2f}%")
    print(f"{'Precision':<25} {trad_metrics['precision']*100:>8.2f}%{' ':>20} {siamese_metrics['precision']*100:>8.2f}%")
    print(f"{'Recall (Sensitivity)':<25} {trad_metrics['recall']*100:>8.2f}%{' ':>20} {siamese_metrics['recall']*100:>8.2f}%")
    print(f"{'Inference Latency':<25} {trad_metrics['latency_ms']:>8.2f} ms/tile{' ':>13} {siamese_metrics['latency_ms']:>8.2f} ms/tile")
    print("=" * 85)

    out_dir = "outputs/module_11"
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics JSON
    metrics_summary = {
        "siamese_unet": siamese_metrics,
        "traditional_delta_ndvi": trad_metrics,
        "parameters": n_params
    }
    with open(os.path.join(out_dir, "siamese_change_detection_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # --------------------------------------------------------------------------
    # Figure 1: Training Convergence Curves
    # --------------------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    epochs_range = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs_range, history["train_loss"], "o-", color="#d32f2f", label="Train Loss", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], "s--", color="#f57c00", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Compound Loss (BCE + Dice)", fontweight="bold")
    ax1.set_title("Siamese Change Detection: Loss Convergence", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs_range, history["train_iou"], "o-", color="#2e7d32", label="Train IoU", linewidth=2)
    ax2.plot(epochs_range, history["val_iou"], "s--", color="#0288d1", label="Val IoU", linewidth=2)
    ax2.plot(epochs_range, history["val_dice"], "^:", color="#7b1fa2", label="Val Dice Score", linewidth=2)
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Overlap Metric (%)", fontweight="bold")
    ax2.set_title("Validation Accuracy & IoU Convergence", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig1_path = os.path.join(out_dir, "01_siamese_training_curves.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  📸 Saved training curves -> '{fig1_path}'")

    # --------------------------------------------------------------------------
    # Figure 2: Visual Comparison: Siamese DL vs. Traditional Delta-NDVI
    # --------------------------------------------------------------------------
    trained_model.eval()
    sample_t1, sample_t2, sample_mask = next(iter(test_loader))
    with torch.no_grad():
        logits, diff_feats = trained_model.forward_with_features(sample_t1[0:1].to(device), sample_t2[0:1].to(device))
        prob_map = torch.sigmoid(logits).cpu().numpy()[0, 0]
        bin_pred = (prob_map >= 0.5).astype(int)

    # Traditional Delta-NDVI on the same sample
    nir1 = sample_t1[0, 3].numpy()
    red1 = sample_t1[0, 2].numpy()
    nir2 = sample_t2[0, 3].numpy()
    red2 = sample_t2[0, 2].numpy()
    ndvi1 = (nir1 - red1) / (nir1 + red1 + 1e-7)
    ndvi2 = (nir2 - red2) / (nir2 + red2 + 1e-7)
    delta_ndvi = ndvi1 - ndvi2
    trad_pred = (delta_ndvi >= 0.50).astype(int)

    gt_mask = sample_mask[0, 0].numpy().astype(int)

    fig2, axes2 = plt.subplots(1, 6, figsize=(22, 4.2), constrained_layout=True)

    axes2[0].imshow(make_rgb(sample_t1[0]))
    axes2[0].set_title("Pre-Disturbance (T1)\nRGB Composite", fontsize=10, fontweight="bold")
    axes2[0].axis("off")

    axes2[1].imshow(make_rgb(sample_t2[0]))
    axes2[1].set_title("Post-Disturbance (T2)\nRGB Composite", fontsize=10, fontweight="bold")
    axes2[1].axis("off")

    axes2[2].imshow(trad_pred, cmap="coolwarm", vmin=0, vmax=1)
    axes2[2].set_title("Traditional Delta-NDVI\nThresholded Change", fontsize=10, fontweight="bold")
    axes2[2].axis("off")

    im_prob = axes2[3].imshow(prob_map, cmap="inferno", vmin=0, vmax=1)
    axes2[3].set_title("Siamese Change Map\n(Probability Heatmap)", fontsize=10, fontweight="bold")
    axes2[3].axis("off")
    plt.colorbar(im_prob, ax=axes2[3], fraction=0.046, pad=0.04)

    axes2[4].imshow(bin_pred, cmap="Reds", vmin=0, vmax=1)
    axes2[4].set_title("Siamese Binary Mask\n(Threshold >= 0.5)", fontsize=10, fontweight="bold")
    axes2[4].axis("off")

    axes2[5].imshow(gt_mask, cmap="Greens", vmin=0, vmax=1)
    axes2[5].set_title("Ground Truth Mask\n(True Deforestation)", fontsize=10, fontweight="bold")
    axes2[5].axis("off")

    fig2_path = os.path.join(out_dir, "02_siamese_vs_traditional_change_detection.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  📸 Saved visual comparison -> '{fig2_path}'")

    # --------------------------------------------------------------------------
    # Figure 3: Multi-Scale Deep Differential Feature Maps
    # --------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)

    diff_names = [
        ("diff_level_1", "Differential Features: Level 1\n(High-Frequency Edge Discrepancies)"),
        ("diff_level_2", "Differential Features: Level 2\n(Mid-Scale Textural Anomalies)"),
        ("diff_level_3", "Differential Features: Level 3\n(Sub-Canopy Structural Shifts)"),
        ("diff_bottleneck", "Differential Features: Bottleneck\n(Macro-Scale Semantic Deforestation)")
    ]

    for i, (key, title) in enumerate(diff_names):
        feat = diff_feats[key]
        feat_resized = F.interpolate(feat, size=(128, 128), mode="bilinear", align_corners=False).cpu().numpy()[0, 0]
        # Normalize for display
        feat_norm = (feat_resized - feat_resized.min()) / (feat_resized.max() - feat_resized.min() + 1e-8)

        im = axes3[i].imshow(feat_norm, cmap="magma")
        axes3[i].set_title(title, fontsize=10, fontweight="bold")
        axes3[i].axis("off")
        plt.colorbar(im, ax=axes3[i], fraction=0.046, pad=0.04)

    fig3_path = os.path.join(out_dir, "03_multiscale_differential_features.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  📸 Saved differential feature activations -> '{fig3_path}'")

    print("\n" + "=" * 80)
    print("✅ MODULE 11: SIAMESE CHANGE DETECTION COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
