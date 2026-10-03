"""
run_module_13.py

Master execution script for Module 13: Deforestation Severity & Post-Disturbance Classification.
Accomplishes:
  1. Multi-temporal satellite imagery ingestion (Bands: Blue, Green, Red, NIR, SWIR)
  2. Computation of USGS spectral disturbance indices: NBR, dNBR, RdNBR, RBR, dNDVI, dNDMI
  3. Continuous disturbance gradient mapping & 4-tier USGS severity classification
  4. Multi-class machine learning classification (Random Forest with 18 features)
  5. Multi-class confusion matrix, precision, recall, and feature importance rankings
  6. Spatial area statistics: Hectares and percentage affected by severity tier
  7. Publication-quality figures in outputs/module_13/
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

from modules.module_13_severity_classification.severity_indices import DisturbanceSeverityCalculator
from modules.module_13_severity_classification.severity_classifier import ForestDisturbanceClassifier


def make_rgb(bands_5ch):
    """Creates normalized RGB composite from (5, H, W) numpy array."""
    def strt(b):
        p2, p98 = np.percentile(b, (2, 98))
        if p98 <= p2:
            return np.clip(b, 0.0, 1.0)
        return np.clip((b - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
    # Red: band 2, Green: band 1, Blue: band 0
    return np.stack([strt(bands_5ch[2]), strt(bands_5ch[1]), strt(bands_5ch[0])], axis=-1)


def main():
    print("\n" + "=" * 80)
    print("🛰️  MODULE 13: DEFORESTATION SEVERITY & POST-DISTURBANCE CLASSIFICATION")
    print("=" * 80)

    # 1. Initialize Indices Calculator & Classifier
    print("\n[1/5] Initializing Disturbance Severity Calculator & ML Classifier...")
    calc = DisturbanceSeverityCalculator()
    clf = ForestDisturbanceClassifier(n_estimators=100, max_depth=12, random_state=42)

    # 2. Ingest Multi-Temporal Satellite Scenes
    print("\n[2/5] Ingesting Multi-Temporal Satellite Scenes...")
    train_before = sorted(glob.glob("dataset/train/before/*.tif"))
    train_after  = sorted(glob.glob("dataset/train/after/*.tif"))
    test_before  = sorted(glob.glob("dataset/test/before/*.tif"))
    test_after   = sorted(glob.glob("dataset/test/after/*.tif"))

    print(f"  • Training scenes : {len(train_before)} scenes")
    print(f"  • Test scenes     : {len(test_before)} scenes")

    # Sample pixel dataset for training
    print("\n[3/5] Extracting 18-Feature Multi-Temporal Vectors for Severity Modeling...")
    X_train_list, y_train_list = [], []

    for b_path, a_path in zip(train_before[:4], train_after[:4]):
        with rasterio.open(b_path) as s1:
            t1 = s1.read()
        with rasterio.open(a_path) as s2:
            t2 = s2.read()

        feats = clf.extract_pixel_features(t1, t2)
        idx_dict = calc.compute_all_indices(t1, t2)
        labels = calc.classify_severity_usgs(idx_dict["dnbr"]).flatten()

        # Stratified subsample 5,000 pixels per scene for fast execution
        np.random.seed(42)
        idx_sample = np.random.choice(len(labels), size=min(5000, len(labels)), replace=False)
        X_train_list.append(feats[idx_sample])
        y_train_list.append(labels[idx_sample])

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    print(f"  • Training pixel samples: {X_train.shape[0]:,} pixels (18 features)")

    # Prepare Test Set
    X_test_list, y_test_list = [], []
    for b_path, a_path in zip(test_before[:2], test_after[:2]):
        with rasterio.open(b_path) as s1:
            t1 = s1.read()
        with rasterio.open(a_path) as s2:
            t2 = s2.read()

        feats = clf.extract_pixel_features(t1, t2)
        idx_dict = calc.compute_all_indices(t1, t2)
        labels = calc.classify_severity_usgs(idx_dict["dnbr"]).flatten()

        idx_sample = np.random.choice(len(labels), size=min(3000, len(labels)), replace=False)
        X_test_list.append(feats[idx_sample])
        y_test_list.append(labels[idx_sample])

    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    print(f"  • Test pixel samples    : {X_test.shape[0]:,} pixels")

    # 3. Train Classifier & Evaluate
    print("\n[4/5] Training Multi-Class Severity Random Forest...")
    clf.fit(X_train, y_train)
    print(f"  • Model trained in {clf.train_time:.2f} seconds.")

    metrics = clf.evaluate(X_test, y_test)
    print("\n" + "=" * 80)
    print("📊 MULTI-CLASS DISTURBANCE SEVERITY BENCHMARK")
    print("=" * 80)
    print(f"  • Overall Test Accuracy : {metrics['accuracy']*100:>6.2f}%")
    print(f"  • Macro-Averaged F1     : {metrics['macro_f1']*100:>6.2f}%")
    print(f"  • Weighted F1-Score     : {metrics['weighted_f1']*100:>6.2f}%")
    print(f"  • Inference Latency     : {metrics['inference_time_sec']*1000 / len(X_test):.3f} ms per pixel")
    print("=" * 80)

    # Inspect Feature Importance
    feat_df = clf.get_feature_importances()
    print("\n🌟 TOP 5 MOST PREDICTIVE SPECTRAL FEATURES:")
    for rank, (_, row) in enumerate(feat_df.head(5).iterrows(), start=1):
        print(f"  #{rank} {row['Feature']:<14}: {row['Importance']*100:>5.2f}% relative importance")

    out_dir = "outputs/module_13"
    os.makedirs(out_dir, exist_ok=True)

    # 4. Process Full Scene Spatial Prediction & Area Stats
    print("\n[5/5] Generating Spatial Severity Maps and Area Statistics...")
    with rasterio.open(test_before[0]) as s1:
        scene_t1 = s1.read()
    with rasterio.open(test_after[0]) as s2:
        scene_t2 = s2.read()

    indices_full = calc.compute_all_indices(scene_t1, scene_t2)
    pred_severity_map = clf.predict_spatial_raster(scene_t1, scene_t2)
    area_stats = calc.compute_severity_area_stats(pred_severity_map, pixel_res_meters=30.0)

    print("\n" + "=" * 85)
    print("🌲 FOREST DISTURBANCE AREA IMPACT AUDIT (SCENE 1)")
    print("=" * 85)
    print(f"{'Severity Tier':<30} {'Pixels':<12} {'Area (ha)':<15} {'Area (km²)':<14} {'Share (%)':<10}")
    print("-" * 85)
    for name, s in area_stats.items():
        print(f"{name:<30} {s['pixel_count']:<12,} {s['area_hectares']:<15,} {s['area_km2']:<14} {s['percentage']:>5.2f}%")
    print("=" * 85)

    # Save Metrics JSON
    summary_data = {
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"]
        },
        "feature_importances": feat_df.to_dict(orient="records"),
        "scene_area_stats": area_stats
    }
    with open(os.path.join(out_dir, "severity_classification_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Figure 1: Spectral Disturbance Indices
    # --------------------------------------------------------------------------
    rgb1 = make_rgb(scene_t1)
    rgb2 = make_rgb(scene_t2)

    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)

    axes1[0, 0].imshow(rgb1)
    axes1[0, 0].set_title("Pre-Disturbance (T1) RGB", fontsize=11, fontweight="bold")
    axes1[0, 0].axis("off")

    axes1[0, 1].imshow(rgb2)
    axes1[0, 1].set_title("Post-Disturbance (T2) RGB", fontsize=11, fontweight="bold")
    axes1[0, 1].axis("off")

    im_dnbr = axes1[0, 2].imshow(indices_full["dnbr"], cmap="RdYlGn_r", vmin=-0.2, vmax=1.0)
    axes1[0, 2].set_title("Differenced NBR (dNBR)", fontsize=11, fontweight="bold")
    axes1[0, 2].axis("off")
    plt.colorbar(im_dnbr, ax=axes1[0, 2], fraction=0.046, pad=0.04)

    im_rdnbr = axes1[1, 0].imshow(indices_full["rdnbr"], cmap="inferno", vmin=0, vmax=1.5)
    axes1[1, 0].set_title("Relativized dNBR (RdNBR)", fontsize=11, fontweight="bold")
    axes1[1, 0].axis("off")
    plt.colorbar(im_rdnbr, ax=axes1[1, 0], fraction=0.046, pad=0.04)

    im_dndvi = axes1[1, 1].imshow(indices_full["dndvi"], cmap="coolwarm", vmin=-0.5, vmax=0.8)
    axes1[1, 1].set_title("Delta NDVI (Canopy Loss)", fontsize=11, fontweight="bold")
    axes1[1, 1].axis("off")
    plt.colorbar(im_dndvi, ax=axes1[1, 1], fraction=0.046, pad=0.04)

    im_dndmi = axes1[1, 2].imshow(indices_full["dndmi"], cmap="BrBG", vmin=-0.5, vmax=0.8)
    axes1[1, 2].set_title("Delta NDMI (Moisture Deficit)", fontsize=11, fontweight="bold")
    axes1[1, 2].axis("off")
    plt.colorbar(im_dndmi, ax=axes1[1, 2], fraction=0.046, pad=0.04)

    fig1_path = os.path.join(out_dir, "01_spectral_severity_indices_dnbr_rdnbr.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  📸 Saved spectral indices figure -> '{fig1_path}'")

    # --------------------------------------------------------------------------
    # Figure 2: Spatial Severity Classification Map & Area Distribution
    # --------------------------------------------------------------------------
    fig2, (ax_map, ax_pie) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    # Custom colormap for severity: Dark Green, Yellow, Orange, Red
    cmap_sev = mcolors.ListedColormap(["#1b5e20", "#fbc02d", "#f57c00", "#d32f2f"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_sev = mcolors.BoundaryNorm(bounds, cmap_sev.N)

    im_sev = ax_map.imshow(pred_severity_map, cmap=cmap_sev, norm=norm_sev)
    ax_map.set_title("Full-Scene Classified Disturbance Severity Map", fontsize=12, fontweight="bold")
    ax_map.axis("off")

    cbar = plt.colorbar(im_sev, ax=ax_map, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["0: Undisturbed", "1: Low Severity", "2: Moderate", "3: High Severity"], fontsize=10, fontweight="bold")

    # Area Pie Chart
    labels = list(area_stats.keys())
    sizes = [s["percentage"] for s in area_stats.values()]
    colors = ["#1b5e20", "#fbc02d", "#f57c00", "#d32f2f"]

    # Filter out 0% classes for clean pie chart
    filtered_labels, filtered_sizes, filtered_colors = [], [], []
    for l, s, c in zip(labels, sizes, colors):
        if s > 0.05:
            filtered_labels.append(f"{l}\n({s:.1f}%)")
            filtered_sizes.append(s)
            filtered_colors.append(c)

    ax_pie.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, startangle=140,
               wedgeprops={"edgecolor": "black", "linewidth": 1.2}, textprops={"fontweight": "bold", "fontsize": 10})
    ax_pie.set_title("Spatial Breakdown of Forest Canopy Disturbance (%)", fontsize=12, fontweight="bold")

    fig2_path = os.path.join(out_dir, "02_spatial_severity_classification_map.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  📸 Saved spatial severity classification map -> '{fig2_path}'")

    # --------------------------------------------------------------------------
    # Figure 3: Confusion Matrix & Feature Importances
    # --------------------------------------------------------------------------
    fig3, (ax_cm, ax_imp) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    cm = metrics["confusion_matrix"]
    cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    im_cm = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_title("Multi-Class Severity Confusion Matrix (Normalized)", fontsize=12, fontweight="bold")
    ax_cm.set_xlabel("Predicted Severity Class", fontsize=11, fontweight="bold")
    ax_cm.set_ylabel("True USGS Severity Class", fontsize=11, fontweight="bold")
    class_ticks = ["0: Undisturbed", "1: Low", "2: Moderate", "3: High"]
    ax_cm.set_xticks(range(4))
    ax_cm.set_yticks(range(4))
    ax_cm.set_xticklabels(class_ticks, fontsize=9, fontweight="bold")
    ax_cm.set_yticklabels(class_ticks, fontsize=9, fontweight="bold")

    for i in range(4):
        for j in range(4):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax_cm.text(j, i, f"{val*100:.1f}%\n({cm[i, j]})", ha="center", va="center", color=color, fontsize=9, fontweight="bold")
    plt.colorbar(im_cm, ax=ax_cm, fraction=0.046, pad=0.04)

    # Feature Importance Bar Chart
    top10 = feat_df.head(10).iloc[::-1]
    ax_imp.barh(top10["Feature"], top10["Importance"] * 100, color="#1976d2", edgecolor="black")
    ax_imp.set_xlabel("Relative Importance (%)", fontsize=11, fontweight="bold")
    ax_imp.set_title("Top 10 Most Predictive Spectral Features", fontsize=12, fontweight="bold")
    ax_imp.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(top10["Importance"] * 100):
        ax_imp.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    fig3_path = os.path.join(out_dir, "03_severity_confusion_matrix_feature_importance.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  📸 Saved confusion matrix & feature importance -> '{fig3_path}'")

    print("\n" + "=" * 80)
    print("✅ MODULE 13: DEFORESTATION SEVERITY CLASSIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
