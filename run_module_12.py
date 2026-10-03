"""
run_module_12.py

Master execution script for Module 12: Deep Learning-Based Deforestation Detection Fusion Strategies.
Comprehensive comparison of:
  1. Early Fusion (EF): 10-band input stacking -> U-Net
  2. Late Fusion (LF): Dual independent encoders -> bottleneck concatenation
  3. Siamese Feature Differencing (Siam-Diff): Dual shared encoder + absolute difference skips
  4. Siamese Feature Concatenation (Siam-Concat): Dual shared encoder + concatenated skips

Outputs:
  - Benchmark Leaderboard: Mean IoU, Dice Score, Parameters, Latency, Train Time
  - outputs/module_12/01_fusion_strategies_comparison.png
  - outputs/module_12/02_spatial_fusion_predictions.png
  - outputs/module_12/03_fusion_error_residuals.png
  - outputs/module_12/fusion_strategies_leaderboard.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

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
from modules.module_12_deforestation_fusion.fusion_models import (
    EarlyFusionUNet, LateFusionUNet, SiameseDiffUNet, SiameseConcatUNet
)
from modules.module_12_deforestation_fusion.fusion_comparator import (
    compare_fusion_strategies, count_parameters
)


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


def main():
    print("\n" + "=" * 80)
    print("🛰️  MODULE 12: DEEP LEARNING FUSION STRATEGIES FOR DEFORESTATION DETECTION")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/5] Compute device: [{device}]")

    # 1. Prepare DataLoaders
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

    # 2. Instantiate 4 Fusion Models
    print("\n[3/5] Instantiating 4 Multi-Temporal Fusion Architectures...")
    models_to_test = {
        "Early Fusion (EF)": EarlyFusionUNet(in_channels_per_image=5, out_channels=1, base_features=16),
        "Late Fusion (LF)": LateFusionUNet(in_channels_per_image=5, out_channels=1, base_features=16),
        "Siamese Differencing": SiameseDiffUNet(in_channels_per_image=5, out_channels=1, base_features=16),
        "Siamese Concatenation": SiameseConcatUNet(in_channels_per_image=5, out_channels=1, base_features=16),
    }

    for name, m in models_to_test.items():
        print(f"  • {name:<22}: {count_parameters(m):>9,} trainable parameters")

    # 3. Train & Compare All Fusion Strategies
    print("\n[4/5] Training and Benchmarking all fusion paradigms (4 epochs each, Adam lr=1e-3)...")
    print("-" * 85)
    leaderboard_df, trained_models = compare_fusion_strategies(
        models_dict=models_to_test,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=4,
        lr=1e-3,
        device=device
    )
    print("-" * 85)

    # Display Leaderboard
    print("\n" + "=" * 95)
    print("🏆 MULTI-TEMPORAL FUSION STRATEGY LEADERBOARD")
    print("=" * 95)
    header = f"{'Rank':<5} {'Fusion Strategy':<24} {'Params':<12} {'IoU (%)':<10} {'Dice (%)':<10} {'Accuracy (%)':<14} {'Latency (ms)':<12}"
    print(header)
    print("-" * 95)
    for rank, (_, row) in enumerate(leaderboard_df.iterrows(), start=1):
        print(f"#{rank:<4} {row['Fusion Strategy']:<24} {row['Parameters']:<12} {row['Mean IoU']*100:>6.2f}%    {row['Dice Score']*100:>6.2f}%    {row['Pixel Accuracy']*100:>8.2f}%       {row['Latency (ms)']:>6.2f} ms")
    print("=" * 95)

    out_dir = "outputs/module_12"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "fusion_strategies_leaderboard.csv")
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"\n  💾 Saved benchmark leaderboard -> '{csv_path}'")

    # 4. Generate Visualizations
    print(f"\n[5/5] Generating Visualizations in '{out_dir}/'...")

    # --------------------------------------------------------------------------
    # Figure 1: Performance Comparison Bar Charts & Trade-Offs
    # --------------------------------------------------------------------------
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    archs = leaderboard_df["Fusion Strategy"].tolist()
    ious = [v * 100 for v in leaderboard_df["Mean IoU"]]
    dices = [v * 100 for v in leaderboard_df["Dice Score"]]
    latencies = leaderboard_df["Latency (ms)"].tolist()
    params_k = [int(p.replace(",", "")) / 1000.0 for p in leaderboard_df["Parameters"]]

    x = np.arange(len(archs))
    width = 0.35

    # Panel 1: IoU & Dice Score
    bars1 = ax1.bar(x - width/2, ious, width, label="Mean IoU (%)", color="#2e7d32", alpha=0.85, edgecolor="black")
    bars2 = ax1.bar(x + width/2, dices, width, label="Dice Score (%)", color="#0288d1", alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Change Detection Overlap (IoU & Dice)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(archs, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 2: Latency per Tile
    bars_lat = ax2.bar(archs, latencies, color="#ff8f00", edgecolor="black", alpha=0.85, width=0.5)
    ax2.set_ylabel("Latency per 128x128 Tile (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("CPU Inference Latency", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(archs, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars_lat:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f} ms", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 3: Parameter Efficiency vs Accuracy
    colors = ["#2e7d32", "#d81b60", "#0288d1", "#8e24aa"]
    for i, arch in enumerate(archs):
        ax3.scatter(params_k[i], ious[i], s=250, color=colors[i % len(colors)], edgecolors="black", linewidth=1.5, zorder=5)
        ax3.annotate(f" {arch}\n ({params_k[i]:.0f}k params)", (params_k[i], ious[i]),
                     xytext=(6, -4), textcoords="offset points", fontsize=9, fontweight="bold")
    ax3.set_xlabel("Trainable Parameters (in Thousands)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Mean IoU (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Parameter Footprint vs Change Detection IoU", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)

    fig1_path = os.path.join(out_dir, "01_fusion_strategies_comparison.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  📸 Saved comparison charts -> '{fig1_path}'")

    # --------------------------------------------------------------------------
    # Figure 2: Spatial Prediction Visualizations across all 4 Fusion Strategies
    # --------------------------------------------------------------------------
    sample_t1, sample_t2, sample_mask = next(iter(test_loader))
    t1_chip = sample_t1[0:1].to(device)
    t2_chip = sample_t2[0:1].to(device)
    gt_mask = sample_mask[0, 0].numpy().astype(int)

    rgb_t1 = make_rgb(sample_t1[0])
    rgb_t2 = make_rgb(sample_t2[0])

    fig2, axes2 = plt.subplots(1, 7, figsize=(24, 3.8), constrained_layout=True)

    axes2[0].imshow(rgb_t1)
    axes2[0].set_title("Pre-Disturbance (T1)\nRGB Composite", fontsize=10, fontweight="bold")
    axes2[0].axis("off")

    axes2[1].imshow(rgb_t2)
    axes2[1].set_title("Post-Disturbance (T2)\nRGB Composite", fontsize=10, fontweight="bold")
    axes2[1].axis("off")

    fusion_keys = ["Early Fusion (EF)", "Late Fusion (LF)", "Siamese Differencing", "Siamese Concatenation"]
    preds_dict = {}

    for idx, f_name in enumerate(fusion_keys, start=2):
        m = trained_models[f_name]
        m.eval()
        with torch.no_grad():
            lgt = m(t1_chip, t2_chip)
            prob = torch.sigmoid(lgt).cpu().numpy()[0, 0]
            pred = (prob >= 0.5).astype(int)
            preds_dict[f_name] = pred

        tp = np.sum((pred == 1) & (gt_mask == 1))
        fp = np.sum((pred == 1) & (gt_mask == 0))
        fn = np.sum((pred == 0) & (gt_mask == 1))
        tile_iou = tp / (tp + fp + fn + 1e-8) * 100.0

        axes2[idx].imshow(pred, cmap="coolwarm", vmin=0, vmax=1)
        axes2[idx].set_title(f"{f_name}\nIoU: {tile_iou:.1f}%", fontsize=10, fontweight="bold")
        axes2[idx].axis("off")

    axes2[6].imshow(gt_mask, cmap="Greens", vmin=0, vmax=1)
    axes2[6].set_title("Ground Truth Mask\n(True Deforestation)", fontsize=10, fontweight="bold")
    axes2[6].axis("off")

    fig2_path = os.path.join(out_dir, "02_spatial_fusion_predictions.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  📸 Saved spatial predictions -> '{fig2_path}'")

    # --------------------------------------------------------------------------
    # Figure 3: Error Residual Decompositions
    # --------------------------------------------------------------------------
    # Colors: TP = Green, FP = Red, FN = Blue, TN = Black
    fig3, axes3 = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)

    for idx, f_name in enumerate(fusion_keys):
        pred = preds_dict[f_name]
        h, w = pred.shape
        error_map = np.zeros((h, w, 3), dtype=np.float32)

        # TP: Green [0.18, 0.8, 0.44]
        tp_mask = (pred == 1) & (gt_mask == 1)
        error_map[tp_mask] = [0.18, 0.80, 0.44]

        # FP: Red [0.91, 0.30, 0.24]
        fp_mask = (pred == 1) & (gt_mask == 0)
        error_map[fp_mask] = [0.91, 0.30, 0.24]

        # FN: Blue [0.20, 0.60, 0.86]
        fn_mask = (pred == 0) & (gt_mask == 1)
        error_map[fn_mask] = [0.20, 0.60, 0.86]

        # TN: Dark gray [0.15, 0.15, 0.15]
        tn_mask = (pred == 0) & (gt_mask == 0)
        error_map[tn_mask] = [0.15, 0.15, 0.15]

        axes3[idx].imshow(error_map)
        axes3[idx].set_title(f"{f_name}\nGreen=TP, Red=FP, Blue=FN", fontsize=10, fontweight="bold")
        axes3[idx].axis("off")

    fig3_path = os.path.join(out_dir, "03_fusion_error_residuals.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  📸 Saved error residual decomposition -> '{fig3_path}'")

    print("\n" + "=" * 80)
    print("✅ MODULE 12: DEEP LEARNING FUSION STRATEGIES COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
