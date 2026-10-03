"""
run_module_07.py

Master execution script for Module 7: Forest vs Non-Forest Classification.
Trains, benchmarks, and visualizes traditional machine learning baselines:
  1. Pixel-level Spectral Feature Extraction (13 features per pixel)
  2. Multi-Model Training: Random Forest vs. SVM vs. XGBoost
  3. Feature Importance Analysis (Gini Impurity / Gain)
  4. Comprehensive Benchmark Leaderboard on Unseen Test Imagery
  5. Publication-quality figures in outputs/module_07/
"""

import os
import sys
import glob
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
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

from modules.module_07_forest_classification.ml_classifier import (
    PixelFeatureExtractor,
    ForestMLClassifierSuite
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

def get_forest_ground_truth(bands: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Derives forest vs non-forest ground truth:
    Healthy forest canopy has high NIR (> 0.40) and low Red (< 0.10).
    If a deforestation mask is provided, deforested pixels are labeled 0 (Non-Forest).
    """
    red = bands[2]
    nir = bands[3]
    ndvi = (nir - red) / (nir + red + 1e-7)
    forest = (ndvi >= 0.50).astype(np.uint8)
    if mask is not None:
        forest[mask == 1] = 0  # Deforested areas are non-forest
    return forest

def main():
    print("\n" + "=" * 75)
    print("[*] MODULE 7: FOREST VS NON-FOREST CLASSIFICATION - MASTER RUNNER")
    print("=" * 75)

    train_files = sorted(glob.glob("dataset/train/before/*.tif"))
    test_files  = sorted(glob.glob("dataset/test/before/*.tif"))

    if not train_files or not test_files:
        print("Error: Train/test imagery not found in dataset/. Please run run_module_03.py first.")
        return

    # 1. Collect training pixels across training scenes
    print(f"\n[1/5] Extracting pixel training samples from {len(train_files)} scenes...")
    X_train_list, y_train_list = [], []

    for f_path in train_files[:6]:  # Sample from first 6 training scenes
        with rasterio.open(f_path) as src:
            b = src.read()
        f_label = get_forest_ground_truth(b)
        X_sub, y_sub = PixelFeatureExtractor.extract_training_sample(b, f_label, sample_size=4000)
        X_train_list.append(X_sub)
        y_train_list.append(y_sub)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    print(f"  • Total Training Pixels   : {len(X_train)} ({np.sum(y_train==1)} Forest, {np.sum(y_train==0)} Non-Forest)")
    print(f"  • Feature Vector Length   : {X_train.shape[1]} spectral features")

    # 2. Extract Test Pixels
    print(f"\n[2/5] Preparing unseen test evaluation set...")
    with rasterio.open(test_files[0]) as src:
        test_bands = src.read()
    test_label = get_forest_ground_truth(test_bands)
    X_test, y_test = PixelFeatureExtractor.extract_training_sample(test_bands, test_label, sample_size=10000)

    # 3. Train Classifier Suite
    print("\n[3/5] Training Classifiers: Random Forest vs. SVM vs. XGBoost...")
    suite = ForestMLClassifierSuite(random_state=42)
    suite.train_all(X_train, y_train)

    # 4. Benchmark Evaluation Leaderboard
    print("\n[4/5] Evaluating Models on Test Pixels...")
    leaderboard = suite.evaluate_all(X_test, y_test)

    print("\n" + "=" * 90)
    print(f"{'MODEL':<18} | {'ACCURACY':<9} | {'PRECISION':<10} | {'RECALL':<8} | {'F1-SCORE':<9} | {'IOU':<7} | {'ROC-AUC':<8} | {'TIME'}")
    print("-" * 90)
    for _, row in leaderboard.iterrows():
        print(f"{row['model_name']:<18} | {row['accuracy']*100:>7.2f}% | {row['precision']*100:>8.2f}% | {row['recall']*100:>6.2f}% | {row['f1_score']*100:>7.2f}% | {row['iou']*100:>5.2f}% | {row['roc_auc']:>7.3f} | {row['train_time_sec']}s")
    print("=" * 90)

    # Feature Importance Analysis
    importances = suite.get_feature_importances()
    print("\n🔍 Top 5 Feature Importances (Random Forest):")
    rf_imp = importances.get("Random Forest")
    if rf_imp is not None:
        for feat, score in rf_imp.head(5).items():
            print(f"  • {feat:<16}: {score*100:.2f}%")

    # 5. Generate Visualizations
    out_dir = "outputs/module_07"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[5/5] Generating Benchmark Figures in '{out_dir}/'...")

    # Plot 1: Model Comparison Bar Chart
    fig1, ax1 = plt.subplots(figsize=(10, 5.2))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "iou"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "IoU"]

    x = np.arange(len(metric_labels))
    width = 0.25
    colors = ["#2ecc71", "#3498db", "#e67e22"]

    for idx, (_, row) in enumerate(leaderboard.iterrows()):
        vals = [row[m] * 100 for m in metrics_to_plot]
        ax1.bar(x + (idx - 1) * width, vals, width, label=row["model_name"], color=colors[idx], edgecolor="black", linewidth=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(80, 103)
    ax1.set_title("Traditional Machine Learning Benchmark Comparison (Module 7)", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

    plot1_path = os.path.join(out_dir, "01_model_comparison_metrics.png")
    plt.savefig(plot1_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"  💾 Saved model comparison chart to: {plot1_path}")

    # Plot 2: Feature Importance Ranking (RF vs XGBoost)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    if "Random Forest" in importances:
        rf_s = importances["Random Forest"].head(10).sort_values(ascending=True)
        axes2[0].barh(rf_s.index, rf_s.values * 100, color="#27ae60", edgecolor="black", linewidth=0.5)
        axes2[0].set_title("Random Forest Feature Importance (MDI)\n[Top Spectral Features]", fontsize=11, fontweight="bold")
        axes2[0].set_xlabel("Relative Importance (%)", fontsize=10)
        axes2[0].grid(True, linestyle=":", alpha=0.5)

    if "XGBoost" in importances:
        xgb_s = importances["XGBoost"].head(10).sort_values(ascending=True)
        axes2[1].barh(xgb_s.index, xgb_s.values * 100, color="#e67e22", edgecolor="black", linewidth=0.5)
        axes2[1].set_title("XGBoost Feature Importance (Gain)\n[Top Spectral Features]", fontsize=11, fontweight="bold")
        axes2[1].set_xlabel("Relative Importance (%)", fontsize=10)
        axes2[1].grid(True, linestyle=":", alpha=0.5)

    fig2.suptitle("Spectral Feature Importance Analysis for Forest Detection", fontsize=13, fontweight="bold", y=1.03)
    plot2_path = os.path.join(out_dir, "02_feature_importances.png")
    plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  💾 Saved feature importances to: {plot2_path}")

    # Plot 3: Full-Scene Predicted Forest Maps
    pred_rf = suite.predict_scene_mask("Random Forest", test_bands)
    pred_xgb = suite.predict_scene_mask("XGBoost", test_bands)

    fig3, axes3 = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)

    axes3[0].imshow(make_rgb(test_bands))
    axes3[0].set_title("1. Test Scene (RGB)", fontsize=11, fontweight="bold")
    axes3[0].axis("off")

    axes3[1].imshow(test_label, cmap="Greens_r")
    axes3[1].set_title("2. Ground-Truth Forest Mask\n[Green = Forest, White = Non-Forest]", fontsize=11, fontweight="bold")
    axes3[1].axis("off")

    axes3[2].imshow(pred_rf, cmap="Greens_r")
    axes3[2].set_title(f"3. Random Forest Prediction\nAcc: {leaderboard.loc[leaderboard['model_name']=='Random Forest', 'accuracy'].values[0]*100:.1f}%", fontsize=11, fontweight="bold", color="darkgreen")
    axes3[2].axis("off")

    axes3[3].imshow(pred_xgb, cmap="Greens_r")
    axes3[3].set_title(f"4. XGBoost Prediction\nAcc: {leaderboard.loc[leaderboard['model_name']=='XGBoost', 'accuracy'].values[0]*100:.1f}%", fontsize=11, fontweight="bold", color="darkorange")
    axes3[3].axis("off")

    fig3.suptitle("Full-Scene Forest Classification Comparison (Module 7)", fontsize=13, fontweight="bold", y=1.03)
    plot3_path = os.path.join(out_dir, "03_ml_classified_forest_maps.png")
    plt.savefig(plot3_path, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"  💾 Saved classified forest maps to: {plot3_path}")

    print("\n" + "=" * 75)
    print("🎯 Module 7 ML Baseline Complete!")
    print(f"  • Top Performing Model : {leaderboard.iloc[0]['model_name']} (F1 = {leaderboard.iloc[0]['f1_score']*100:.2f}%)")
    print(f"📁 All visual products saved in: {os.path.abspath(out_dir)}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
