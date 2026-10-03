"""
run_module_10.py

Master execution script for Module 10: Advanced Segmentation Architectures.
Comprehensive comparison of:
  1. Standard U-Net (Ronneberger et al., 2015)
  2. Attention U-Net (Oktay et al., 2018 - Attention Gates on Skips)
  3. DeepLabV3-Lite (Chen et al., 2017 - Atrous Spatial Pyramid Pooling ASPP)
  4. Nested U-Net++ (Zhou et al., 2018 - Dense Nested Skip Pathways)

Outputs:
  - Benchmark Leaderboard: Mean IoU, Dice Score, Parameters, Latency, Train Time
  - outputs/module_10/01_model_comparison_iou_dice.png
  - outputs/module_10/02_attention_gate_visualization.png
  - outputs/module_10/03_architecture_predictions_comparison.png
  - outputs/module_10/model_comparison_leaderboard.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
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

from modules.module_09_unet_segmentation.unet_model import ForestUNet
from modules.module_09_unet_segmentation.segmentation_dataset import create_segmentation_dataloaders
from modules.module_10_advanced_segmentation.advanced_models import AttentionUNet, DeepLabV3Lite, NestedUNetLite
from modules.module_10_advanced_segmentation.model_comparator import compare_segmentation_models, count_parameters


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
    print("🛰️  MODULE 10: ADVANCED SEGMENTATION ARCHITECTURES - MASTER RUNNER")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/5] Compute device: [{device}]")

    # 1. Prepare DataLoaders
    print("\n[2/5] Preparing Dense Segmentation DataLoaders (128x128 tiles)...")
    train_loader, val_loader, test_loader = create_segmentation_dataloaders(
        dataset_root="dataset",
        tile_size=128,
        batch_size=8,
        max_train_tiles=24,
        max_val_tiles=8
    )
    print(f"  • Train tiles : {len(train_loader.dataset)} tiles ({len(train_loader)} batches)")
    print(f"  • Val tiles   : {len(val_loader.dataset)} tiles ({len(val_loader)} batches)")
    print(f"  • Test tiles  : {len(test_loader.dataset)} tiles ({len(test_loader)} batches)")

    # 2. Instantiate Candidate Architectures
    print("\n[3/5] Instantiating 4 Deep Segmentation Architectures...")
    models_to_test = {
        "Standard U-Net": ForestUNet(in_channels=5, out_channels=1, base_features=16),
        "Attention U-Net": AttentionUNet(in_channels=5, out_channels=1, base_features=16),
        "DeepLabV3-Lite": DeepLabV3Lite(in_channels=5, out_channels=1, base_features=16),
        "Nested U-Net++": NestedUNetLite(in_channels=5, out_channels=1, base_features=16),
    }

    for name, m in models_to_test.items():
        print(f"  • {name:<18}: {count_parameters(m):>9,} trainable parameters")

    # 3. Train & Compare Models
    print("\n[4/5] Training and Benchmarking all models (4 epochs each, Adam lr=1e-3)...")
    print("-" * 80)
    leaderboard_df, trained_models = compare_segmentation_models(
        models_dict=models_to_test,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=4,
        lr=1e-3,
        device=device
    )
    print("-" * 80)

    # Display Leaderboard
    print("\n" + "=" * 90)
    print("🏆 SEGMENTATION ARCHITECTURE LEADERBOARD")
    print("=" * 90)
    header = f"{'Rank':<5} {'Architecture':<18} {'Params':<12} {'IoU (%)':<10} {'Dice (%)':<10} {'Accuracy (%)':<14} {'Latency (ms)':<12}"
    print(header)
    print("-" * 90)
    for rank, (_, row) in enumerate(leaderboard_df.iterrows(), start=1):
        print(f"#{rank:<4} {row['Architecture']:<18} {row['Parameters']:<12} {row['Mean IoU']*100:>6.2f}%    {row['Dice Score']*100:>6.2f}%    {row['Pixel Accuracy']*100:>8.2f}%       {row['Latency (ms)']:>6.2f} ms")
    print("=" * 90)

    # Save CSV
    out_dir = "outputs/module_10"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "model_comparison_leaderboard.csv")
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"\n  💾 Saved benchmark leaderboard -> '{csv_path}'")

    # 4. Generate Visualizations
    print(f"\n[5/5] Generating Visualizations in '{out_dir}/'...")

    # --------------------------------------------------------------------------
    # Figure 1: Comparative Bar Charts & Efficiency Trade-offs
    # --------------------------------------------------------------------------
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    archs = leaderboard_df["Architecture"].tolist()
    ious = [v * 100 for v in leaderboard_df["Mean IoU"]]
    dices = [v * 100 for v in leaderboard_df["Dice Score"]]
    latencies = leaderboard_df["Latency (ms)"].tolist()
    params_k = [int(p.replace(",", "")) / 1000.0 for p in leaderboard_df["Parameters"]]

    x = np.arange(len(archs))
    width = 0.35

    # Panel 1: IoU vs Dice Score
    bars1 = ax1.bar(x - width/2, ious, width, label="Mean IoU (%)", color="#2e7d32", alpha=0.85, edgecolor="black")
    bars2 = ax1.bar(x + width/2, dices, width, label="Dice Score (%)", color="#0288d1", alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Segmentation Accuracy: Mean IoU & Dice Score", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(archs, rotation=15, ha="right", fontsize=10, fontweight="bold")
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

    # Panel 2: Inference Latency
    bars_lat = ax2.bar(archs, latencies, color="#ff8f00", edgecolor="black", alpha=0.85, width=0.5)
    ax2.set_ylabel("Latency per 128x128 Tile (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Inference Latency on CPU", fontsize=12, fontweight="bold")
    ax2.set_xticklabels(archs, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars_lat:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f} ms", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 3: Parameter Efficiency (Params vs IoU)
    colors = ["#2e7d32", "#0288d1", "#d81b60", "#8e24aa"]
    for i, arch in enumerate(archs):
        ax3.scatter(params_k[i], ious[i], s=260, color=colors[i % len(colors)], edgecolors="black", linewidth=1.5, zorder=5)
        ax3.annotate(f" {arch}\n ({params_k[i]:.0f}k params)", (params_k[i], ious[i]),
                     xytext=(6, -4), textcoords="offset points", fontsize=9, fontweight="bold")
    ax3.set_xlabel("Trainable Parameters (in Thousands)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Mean IoU (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Parameter Efficiency vs Accuracy", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)

    fig1_path = os.path.join(out_dir, "01_model_comparison_iou_dice.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  📸 Saved comparison charts -> '{fig1_path}'")

    # --------------------------------------------------------------------------
    # Figure 2: Attention Gate Spatial Visualizations
    # --------------------------------------------------------------------------
    att_model = trained_models["Attention U-Net"]
    att_model.eval()

    sample_x, sample_y = next(iter(test_loader))
    with torch.no_grad():
        x_tile = sample_x[0:1].to(device)
        y_tile = sample_y[0:1].numpy()[0, 0]
        logits, att_maps = att_model.forward_with_attention_maps(x_tile)
        prob_map = torch.sigmoid(logits).cpu().numpy()[0, 0]

    rgb_chip = make_rgb(sample_x[0])

    fig2, axes = plt.subplots(1, 6, figsize=(22, 4.2), constrained_layout=True)

    axes[0].imshow(rgb_chip)
    axes[0].set_title("Input RGB Composite\n(Bands 3, 2, 1)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(y_tile, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask\n(1=Forest, 0=Non-Forest)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Attention Gate Levels (Resize to 128x128 for clear visualization)
    a1_map = F.interpolate(att_maps["Level 1"], size=(128, 128), mode="bilinear", align_corners=False).cpu().numpy()[0, 0]
    a2_map = F.interpolate(att_maps["Level 2"], size=(128, 128), mode="bilinear", align_corners=False).cpu().numpy()[0, 0]
    a3_map = F.interpolate(att_maps["Level 3"], size=(128, 128), mode="bilinear", align_corners=False).cpu().numpy()[0, 0]

    im_a1 = axes[2].imshow(a1_map, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Attention Gate 1\n(Fine Boundary Focus)", fontsize=10, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im_a1, ax=axes[2], fraction=0.046, pad=0.04)

    im_a2 = axes[3].imshow(a2_map, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("Attention Gate 2\n(Mid-Level Spatial Context)", fontsize=10, fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im_a2, ax=axes[3], fraction=0.046, pad=0.04)

    im_a3 = axes[4].imshow(a3_map, cmap="magma", vmin=0, vmax=1)
    axes[4].set_title("Attention Gate 3\n(Semantic Canopy Gating)", fontsize=10, fontweight="bold")
    axes[4].axis("off")
    plt.colorbar(im_a3, ax=axes[4], fraction=0.046, pad=0.04)

    im_p = axes[5].imshow(prob_map, cmap="viridis", vmin=0, vmax=1)
    axes[5].set_title("Attention U-Net Output\n(Predicted Forest Probability)", fontsize=10, fontweight="bold")
    axes[5].axis("off")
    plt.colorbar(im_p, ax=axes[5], fraction=0.046, pad=0.04)

    fig2_path = os.path.join(out_dir, "02_attention_gate_visualization.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  📸 Saved attention gate visualization -> '{fig2_path}'")

    # --------------------------------------------------------------------------
    # Figure 3: Side-by-Side Predictions Across All 4 Architectures
    # --------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 6, figsize=(22, 7.5), constrained_layout=True)

    # We evaluate two different sample chips from test_loader
    for row_idx, chip_idx in enumerate([0, min(1, len(sample_x) - 1)]):
        x_chip = sample_x[chip_idx:chip_idx+1].to(device)
        gt_chip = sample_y[chip_idx:chip_idx+1].numpy()[0, 0]
        rgb = make_rgb(sample_x[chip_idx])

        axes3[row_idx, 0].imshow(rgb)
        axes3[row_idx, 0].set_title(f"Test Tile #{chip_idx+1} RGB", fontsize=10, fontweight="bold")
        axes3[row_idx, 0].axis("off")

        axes3[row_idx, 1].imshow(gt_chip, cmap="Greens", vmin=0, vmax=1)
        axes3[row_idx, 1].set_title("Ground Truth Mask", fontsize=10, fontweight="bold")
        axes3[row_idx, 1].axis("off")

        col_offset = 2
        for arch_name in ["Standard U-Net", "Attention U-Net", "DeepLabV3-Lite", "Nested U-Net++"]:
            m = trained_models[arch_name]
            m.eval()
            with torch.no_grad():
                lgt = m(x_chip)
                p_bin = (torch.sigmoid(lgt) >= 0.5).cpu().numpy()[0, 0].astype(int)

            # Compute tile IoU
            tp = np.sum((p_bin == 1) & (gt_chip == 1))
            fp = np.sum((p_bin == 1) & (gt_chip == 0))
            fn = np.sum((p_bin == 0) & (gt_chip == 1))
            tile_iou = tp / (tp + fp + fn + 1e-8) * 100.0

            ax = axes3[row_idx, col_offset]
            ax.imshow(p_bin, cmap="coolwarm", vmin=0, vmax=1)
            ax.set_title(f"{arch_name}\nIoU: {tile_iou:.1f}%", fontsize=10, fontweight="bold")
            ax.axis("off")
            col_offset += 1

    fig3_path = os.path.join(out_dir, "03_architecture_predictions_comparison.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  📸 Saved architecture prediction comparisons -> '{fig3_path}'")

    print("\n" + "=" * 80)
    print("✅ MODULE 10: ADVANCED SEGMENTATION ARCHITECTURES COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
