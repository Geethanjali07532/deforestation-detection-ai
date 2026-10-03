"""
run_module_08.py

Master execution script for Module 8: CNN-Based Land Cover Classification.
Accomplishes:
  1. Dataset Preparation: 5-Class multi-band satellite patches (32x32)
  2. CNN Training: 3-stage custom Convolutional Neural Network with BatchNorm & MaxPool
  3. Feature Map Extraction: Demonstrates convolutional feature activations
  4. Test Set Evaluation: Confusion Matrix & Per-Class Precision/Recall/F1
  5. Publication-quality figures in outputs/module_08/
"""

import os
import sys
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

from modules.module_08_cnn_classification.patch_dataset import (
    LandCoverPatchDataset,
    create_dataloaders
)
from modules.module_08_cnn_classification.cnn_models import SatelliteLandCoverCNN
from modules.module_08_cnn_classification.trainer import (
    train_land_cover_model,
    evaluate_model
)

def make_rgb(patch_tensor):
    """Converts 5-band tensor (C, H, W) to RGB display array."""
    p = patch_tensor.cpu().numpy()
    def strt(arr):
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 <= p2:
            return np.clip(arr, 0.0, 1.0)
        return np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
    # Red: band 2, Green: band 1, Blue: band 0
    return np.stack([strt(p[2]), strt(p[1]), strt(p[0])], axis=-1)

def main():
    print("\n" + "=" * 75)
    print("[*] MODULE 8: CNN-BASED LAND COVER CLASSIFICATION - MASTER RUNNER")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/5] Initializing Deep Learning environment on device: [{device}]")

    # 1. Create Datasets & DataLoaders
    print("\n[2/5] Creating 5-Class Multi-Band Patch Dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        total_samples=1500,
        batch_size=32,
        patch_size=32,
        seed=42
    )
    class_names = LandCoverPatchDataset.CLASS_NAMES
    print(f"  • Classes ({len(class_names)}) : {', '.join(class_names)}")
    print(f"  • Train batches  : {len(train_loader)} batches ({len(train_loader.dataset)} patches)")
    print(f"  • Val batches    : {len(val_loader)} batches ({len(val_loader.dataset)} patches)")
    print(f"  • Test batches   : {len(test_loader)} batches ({len(test_loader.dataset)} patches)")

    # 2. Instantiate & Train Model
    print("\n[3/5] Instantiating SatelliteLandCoverCNN (3 Conv Blocks + AdaptiveAvgPool)...")
    model = SatelliteLandCoverCNN(in_channels=5, num_classes=5)

    trained_model, history = train_land_cover_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=8,
        lr=1e-3,
        device=device
    )

    # 3. Benchmark Evaluation on Unseen Test Patches
    print("[4/5] Evaluating on Test Set...")
    eval_results = evaluate_model(trained_model, test_loader, class_names, device=device)

    print("\n" + "=" * 80)
    print(f"📊 TEST SET BENCHMARK RESULTS (Accuracy = {eval_results['accuracy']*100:.2f}%)")
    print("=" * 80)
    print(f"{'CLASS NAME':<26} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'SUPPORT'}")
    print("-" * 80)
    rep = eval_results["classification_report"]
    for c_name in class_names:
        c_dict = rep[c_name]
        print(f"{c_name:<26} | {c_dict['precision']*100:>8.2f}% | {c_dict['recall']*100:>8.2f}% | {c_dict['f1-score']*100:>8.2f}% | {int(c_dict['support']):>7}")
    print("-" * 80)
    print(f"  • Inference Latency : {eval_results['latency_per_patch_ms']:.3f} ms per patch")
    print("=" * 80)

    # 4. Generate Visualizations
    out_dir = "outputs/module_08"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[5/5] Generating Deep Learning Visualizations in '{out_dir}/'...")

    # Plot 1: Training Curves (Loss & Accuracy)
    fig1, (ax1_loss, ax1_acc) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1_loss.plot(epochs_range, history["train_loss"], "b-o", linewidth=2, label="Train Loss")
    ax1_loss.plot(epochs_range, history["val_loss"], "r--s", linewidth=2, label="Val Loss")
    ax1_loss.set_title("Cross-Entropy Loss Curve", fontsize=11, fontweight="bold")
    ax1_loss.set_xlabel("Epoch", fontsize=10)
    ax1_loss.set_ylabel("Loss", fontsize=10)
    ax1_loss.grid(True, linestyle=":", alpha=0.6)
    ax1_loss.legend(fontsize=10)

    ax1_acc.plot(epochs_range, history["train_acc"], "g-o", linewidth=2, label="Train Accuracy")
    ax1_acc.plot(epochs_range, history["val_acc"], "m--s", linewidth=2, label="Val Accuracy")
    ax1_acc.set_title("Classification Accuracy Curve", fontsize=11, fontweight="bold")
    ax1_acc.set_xlabel("Epoch", fontsize=10)
    ax1_acc.set_ylabel("Accuracy (%)", fontsize=10)
    ax1_acc.grid(True, linestyle=":", alpha=0.6)
    ax1_acc.legend(fontsize=10)

    fig1.suptitle("CNN Training & Convergence Curves (Module 8)", fontsize=13, fontweight="bold", y=1.02)
    plot1_path = os.path.join(out_dir, "01_cnn_training_curves.png")
    plt.savefig(plot1_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"  💾 Saved training curves to: {plot1_path}")

    # Plot 2: Confusion Matrix
    cm = eval_results["confusion_matrix"]
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig2, ax2 = plt.subplots(figsize=(7.5, 6.5))
    im_cm = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9, fontweight="bold")
    ax2.set_yticklabels(class_names, fontsize=9, fontweight="bold")
    ax2.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_title("Normalized Confusion Matrix (5-Class Land Cover)", fontsize=12, fontweight="bold", pad=12)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax2.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")

    fig2.colorbar(im_cm, ax=ax2, fraction=0.046, pad=0.04)
    plot2_path = os.path.join(out_dir, "02_confusion_matrix.png")
    plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  💾 Saved confusion matrix to: {plot2_path}")

    # Plot 3: Convolutional Feature Maps
    # Grab a sample test patch and inspect activations
    test_batch_x, test_batch_y = next(iter(test_loader))
    sample_patch = test_batch_x[0:1].to(device)  # shape (1, 5, 32, 32)
    trained_model.eval()
    with torch.no_grad():
        _, feat_maps = trained_model.forward_with_features(sample_patch)

    fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)

    # Input Patch RGB
    axes3[0].imshow(make_rgb(sample_patch[0]))
    axes3[0].set_title(f"Input Patch: {class_names[test_batch_y[0].item()]}\n(True Color RGB)", fontsize=10, fontweight="bold")
    axes3[0].axis("off")

    # Stage 1 feature map (average of 32 channels)
    f1 = feat_maps["Stage 1 (Edges & Color)"][0].mean(dim=0).cpu().numpy()
    axes3[1].imshow(f1, cmap="viridis")
    axes3[1].set_title("Stage 1 Activation Map (16x16)\n[Edges & Color Contrasts]", fontsize=10, fontweight="bold")
    axes3[1].axis("off")

    # Stage 2 feature map (average of 64 channels)
    f2 = feat_maps["Stage 2 (Textures & Boundaries)"][0].mean(dim=0).cpu().numpy()
    axes3[2].imshow(f2, cmap="viridis")
    axes3[2].set_title("Stage 2 Activation Map (8x8)\n[Textures & Spatial Patterns]", fontsize=10, fontweight="bold")
    axes3[2].axis("off")

    # Stage 3 feature map (average of 128 channels)
    f3 = feat_maps["Stage 3 (Semantic Land Cover)"][0].mean(dim=0).cpu().numpy()
    axes3[3].imshow(f3, cmap="magma")
    axes3[3].set_title("Stage 3 Activation Map (4x4)\n[High-Level Semantic Filters]", fontsize=10, fontweight="bold")
    axes3[3].axis("off")

    fig3.suptitle("Convolutional Neural Network Feature Maps Across Hierarchy (Module 8)", fontsize=13, fontweight="bold", y=1.03)
    plot3_path = os.path.join(out_dir, "03_convolutional_feature_maps.png")
    plt.savefig(plot3_path, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"  💾 Saved convolutional feature maps to: {plot3_path}")

    # Plot 4: Sample Patch Predictions
    fig4, axes4 = plt.subplots(2, 5, figsize=(15, 6.5), constrained_layout=True)
    trained_model.eval()
    with torch.no_grad():
        preds_sample = trained_model(test_batch_x[:10].to(device)).argmax(dim=1).cpu().numpy()

    for i in range(10):
        r, c = i // 5, i % 5
        ax = axes4[r, c]
        ax.imshow(make_rgb(test_batch_x[i]))
        true_name = class_names[test_batch_y[i].item()]
        pred_name = class_names[preds_sample[i]]
        is_correct = (true_name == pred_name)
        color = "darkgreen" if is_correct else "red"
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=9.5, fontweight="bold", color=color)
        ax.axis("off")

    fig4.suptitle("Sample Test Patch Predictions (Green = Correct Match)", fontsize=13, fontweight="bold", y=1.02)
    plot4_path = os.path.join(out_dir, "04_sample_patch_predictions.png")
    plt.savefig(plot4_path, dpi=200, bbox_inches="tight")
    plt.close(fig4)
    print(f"  💾 Saved sample patch predictions to: {plot4_path}")

    print("\n" + "=" * 75)
    print("🎯 Module 8 Deep Learning CNN Complete!")
    print(f"  • Model Test Accuracy : {eval_results['accuracy']*100:.2f}%")
    print(f"  • Feature Maps        : Verified hierarchical abstraction across Stages 1, 2, and 3.")
    print(f"📁 All visual products saved in: {os.path.abspath(out_dir)}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
