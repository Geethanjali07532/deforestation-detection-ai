"""
run_module_09.py

Master execution script for Module 9: Forest Segmentation Using U-Net.
Accomplishes:
  1. Dense Semantic Segmentation Dataset Setup
  2. ForestUNet Architecture Initialization (Encoder-Decoder with Skip Connections)
  3. Compound Loss Optimization: Combined BCE + Dice Loss
  4. Test Set Evaluation: IoU (Jaccard Index), Dice Score, Precision, Recall
  5. Dense Full-Scene Forest Mask Prediction
  6. Publication-quality figures in outputs/module_09/
"""

import os
import sys
import numpy as np
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

from modules.module_09_unet_segmentation.unet_model import ForestUNet
from modules.module_09_unet_segmentation.segmentation_dataset import create_segmentation_dataloaders
from modules.module_09_unet_segmentation.trainer import train_unet, evaluate_unet

def make_rgb(tensor_5ch):
    """Creates RGB composite from (5, H, W) tensor."""
    arr = tensor_5ch.cpu().numpy()
    def strt(b):
        p2, p98 = np.percentile(b, (2, 98))
        if p98 <= p2:
            return np.clip(b, 0.0, 1.0)
        return np.clip((b - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
    # Red: band 2, Green: band 1, Blue: band 0
    return np.stack([strt(arr[2]), strt(arr[1]), strt(arr[0])], axis=-1)

def main():
    print("\n" + "=" * 75)
    print("[*] MODULE 9: FOREST SEGMENTATION USING U-NET - MASTER RUNNER")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/5] Initializing Deep Semantic Segmentation on device: [{device}]")

    # 1. Prepare DataLoaders
    print("\n[2/5] Preparing Dense Segmentation DataLoaders (128x128 tiles)...")
    train_loader, val_loader, test_loader = create_segmentation_dataloaders(
        dataset_root="dataset",
        tile_size=128,
        batch_size=8
    )
    print(f"  • Train tiles : {len(train_loader.dataset)} tiles ({len(train_loader)} batches)")
    print(f"  • Val tiles   : {len(val_loader.dataset)} tiles ({len(val_loader)} batches)")
    print(f"  • Test tiles  : {len(test_loader.dataset)} tiles ({len(test_loader)} batches)")

    # 2. Instantiate & Train U-Net
    print("\n[3/5] Instantiating ForestUNet (Encoder-Decoder with Skip Connections)...")
    model = ForestUNet(in_channels=5, out_channels=1, base_features=16)

    trained_model, history = train_unet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        lr=1e-3,
        device=device
    )

    # 3. Benchmark Evaluation on Unseen Test Imagery
    print("[4/5] Evaluating U-Net Semantic Segmentation on Test Set...")
    metrics = evaluate_unet(trained_model, test_loader, device=device)

    print("\n" + "=" * 80)
    print(f"📊 U-NET TEST SET BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  • Mean IoU (Jaccard Index) : {metrics['iou'] * 100:>6.2f}%")
    print(f"  • Dice Score (F1-Score)    : {metrics['f1_dice'] * 100:>6.2f}%")
    print(f"  • Pixel Accuracy           : {metrics['accuracy'] * 100:>6.2f}%")
    print(f"  • Precision                : {metrics['precision'] * 100:>6.2f}%")
    print(f"  • Recall (Sensitivity)     : {metrics['recall'] * 100:>6.2f}%")
    print(f"  • Inference Latency        : {metrics['latency_per_tile_ms']:.2f} ms per 128x128 tile")
    print("=" * 80)

    # 4. Generate Visualizations
    out_dir = "outputs/module_09"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[5/5] Generating Segmentation Figures in '{out_dir}/'...")

    # Plot 1: Training Convergence Curves (Loss, IoU, Dice)
    fig1, (ax1_loss, ax1_iou) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1_loss.plot(epochs_range, history["train_loss"], "b-o", linewidth=2, label="Train Loss (BCE+Dice)")
    ax1_loss.plot(epochs_range, history["val_loss"], "r--s", linewidth=2, label="Val Loss")
    ax1_loss.set_title("Combined BCE + Dice Loss", fontsize=11, fontweight="bold")
    ax1_loss.set_xlabel("Epoch", fontsize=10)
    ax1_loss.set_ylabel("Loss", fontsize=10)
    ax1_loss.grid(True, linestyle=":", alpha=0.6)
    ax1_loss.legend(fontsize=10)

    ax1_iou.plot(epochs_range, history["train_iou"], "g-o", linewidth=2, label="Train IoU (%)")
    ax1_iou.plot(epochs_range, history["val_iou"], "m--s", linewidth=2, label="Val IoU (%)")
    ax1_iou.plot(epochs_range, history["val_dice"], "c-.^", linewidth=2, label="Val Dice (%)")
    ax1_iou.set_title("Segmentation Overlap (IoU & Dice)", fontsize=11, fontweight="bold")
    ax1_iou.set_xlabel("Epoch", fontsize=10)
    ax1_iou.set_ylabel("Score (%)", fontsize=10)
    ax1_iou.grid(True, linestyle=":", alpha=0.6)
    ax1_iou.legend(fontsize=10)

    fig1.suptitle("U-Net Semantic Segmentation Convergence Curves (Module 9)", fontsize=13, fontweight="bold", y=1.02)
    plot1_path = os.path.join(out_dir, "01_unet_training_curves.png")
    plt.savefig(plot1_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"  💾 Saved training curves to: {plot1_path}")

    # Plot 2: Full-Scene Dense Forest Segmentation Prediction
    test_batch_x, test_batch_y = next(iter(test_loader))
    sample_x = test_batch_x[0:1].to(device)
    sample_gt = test_batch_y[0, 0].numpy()

    trained_model.eval()
    with torch.no_grad():
        logits = trained_model(sample_x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_bin = (probs >= 0.5).astype(np.float32)

    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)

    # 1. RGB
    axes2[0].imshow(make_rgb(sample_x[0]))
    axes2[0].set_title("1. Input Satellite Image (RGB)", fontsize=11, fontweight="bold")
    axes2[0].axis("off")

    # 2. Ground Truth
    axes2[1].imshow(sample_gt, cmap="Greens_r")
    axes2[1].set_title("2. Ground-Truth Forest Mask\n[Forest=1 (Green), Non-Forest=0]", fontsize=11, fontweight="bold")
    axes2[1].axis("off")

    # 3. U-Net Probability Map
    im_p = axes2[2].imshow(probs, cmap="viridis", vmin=0, vmax=1)
    axes2[2].set_title("3. U-Net Sigmoid Confidence Map\n[Dense Pixel Probability]", fontsize=11, fontweight="bold")
    axes2[2].axis("off")
    fig2.colorbar(im_p, ax=axes2[2], fraction=0.046, pad=0.04)

    # 4. Final Binary Prediction
    axes2[3].imshow(pred_bin, cmap="Greens_r")
    axes2[3].set_title(f"4. Binary Prediction (Threshold=0.5)\n[IoU: {metrics['iou']*100:.1f}% | Dice: {metrics['f1_dice']*100:.1f}%]", fontsize=11, fontweight="bold", color="darkgreen")
    axes2[3].axis("off")

    fig2.suptitle("U-Net Dense Forest Segmentation Results (Module 9)", fontsize=13, fontweight="bold", y=1.03)
    plot2_path = os.path.join(out_dir, "02_forest_segmentation_prediction.png")
    plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  💾 Saved segmentation predictions to: {plot2_path}")

    # Plot 3: Architecture Diagram & Skip Connections
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.text(0.1, 0.8, "Encoder Stage 1 (32ch)", style='italic', bbox={'facecolor': 'lightblue', 'alpha': 0.8, 'pad': 8})
    ax3.text(0.1, 0.5, "Encoder Stage 2 (64ch)", style='italic', bbox={'facecolor': 'lightblue', 'alpha': 0.8, 'pad': 8})
    ax3.text(0.1, 0.2, "Encoder Stage 3 (128ch)", style='italic', bbox={'facecolor': 'lightblue', 'alpha': 0.8, 'pad': 8})
    ax3.text(0.38, 0.05, "Bottleneck (256ch)", style='italic', bbox={'facecolor': 'orange', 'alpha': 0.8, 'pad': 8})

    ax3.text(0.65, 0.2, "Decoder Stage 3 (128ch)", style='italic', bbox={'facecolor': 'lightgreen', 'alpha': 0.8, 'pad': 8})
    ax3.text(0.65, 0.5, "Decoder Stage 2 (64ch)", style='italic', bbox={'facecolor': 'lightgreen', 'alpha': 0.8, 'pad': 8})
    ax3.text(0.65, 0.8, "Decoder Stage 1 (32ch)", style='italic', bbox={'facecolor': 'lightgreen', 'alpha': 0.8, 'pad': 8})

    # Arrows for skip connections
    ax3.annotate("", xy=(0.63, 0.83), xytext=(0.33, 0.83), arrowprops=dict(arrowstyle="->", color="purple", lw=2.5, ls="--"))
    ax3.text(0.40, 0.86, "Skip Connection (Level 1)", color="purple", fontweight="bold")

    ax3.annotate("", xy=(0.63, 0.53), xytext=(0.33, 0.53), arrowprops=dict(arrowstyle="->", color="purple", lw=2.5, ls="--"))
    ax3.text(0.40, 0.56, "Skip Connection (Level 2)", color="purple", fontweight="bold")

    ax3.annotate("", xy=(0.63, 0.23), xytext=(0.35, 0.23), arrowprops=dict(arrowstyle="->", color="purple", lw=2.5, ls="--"))
    ax3.text(0.40, 0.26, "Skip Connection (Level 3)", color="purple", fontweight="bold")

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.set_title("U-Net Architecture: Encoder-Decoder Path with High-Resolution Skip Connections", fontsize=12, fontweight="bold")

    plot3_path = os.path.join(out_dir, "03_unet_architecture_flow.png")
    plt.savefig(plot3_path, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"  💾 Saved U-Net architecture diagram to: {plot3_path}")

    print("\n" + "=" * 75)
    print("🎯 Module 9 U-Net Forest Segmentation Complete!")
    print(f"  • Test Mean IoU : {metrics['iou']*100:.2f}%")
    print(f"  • Test Dice     : {metrics['f1_dice']*100:.2f}%")
    print(f"📁 All visual products saved in: {os.path.abspath(out_dir)}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
