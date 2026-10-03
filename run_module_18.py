"""
run_module_18.py

Master execution script for Module 18: Model Evaluation, Robustness & Uncertainty Quantification.
Accomplishes:
  1. Bayesian Monte Carlo Dropout (MCDO) for epistemic uncertainty quantification (Gal & Ghahramani, 2016)
  2. Spatial uncertainty decomposition: Bayesian Mean Probability vs. Epistemic Variance (σ)
  3. Atmospheric & sensor noise stress testing (Gaussian Noise & Cloud Haze attenuation curves)
  4. Expected Calibration Error (ECE) and binned reliability diagram
  5. Publication-grade figures in outputs/module_18/
"""

import os
import sys
import json
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
from modules.module_11_siamese_change_detection.losses import CompoundChangeLoss
from modules.module_18_uncertainty_robustness.mc_dropout_evaluator import (
    MCDropoutChangeDetector, estimate_epistemic_uncertainty
)
from modules.module_18_uncertainty_robustness.robustness_stress_tester import (
    RobustnessStressTester, compute_calibration_curve
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
    print("🛰️  MODULE 18: MODEL EVALUATION, ROBUSTNESS & UNCERTAINTY QUANTIFICATION")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/5] Compute device: [{device}]")

    # 1. Prepare DataLoaders
    print("\n[2/5] Preparing Paired Multi-Temporal DataLoaders...")
    train_loader, val_loader, test_loader = create_siamese_dataloaders(
        dataset_root="dataset",
        tile_size=128,
        batch_size=8,
        max_train_tiles=32,
        max_val_tiles=8
    )

    # 2. Train MCDropout Model
    print("\n[3/5] Instantiating & Training MCDropoutChangeDetector (Dropout p=0.25)...")
    model = MCDropoutChangeDetector(in_channels=5, out_channels=1, base_features=16, dropout_p=0.25).to(device)
    criterion = CompoundChangeLoss(alpha=0.5, pos_weight=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(4):
        model.train()
        for t1_b, t2_b, m_b in train_loader:
            t1_b, t2_b, m_b = t1_b.to(device), t2_b.to(device), m_b.to(device)
            optimizer.zero_grad()
            out = model(t1_b, t2_b)
            loss = criterion(out, m_b)
            loss.backward()
            optimizer.step()
    print(f"  • Trained model with spatial dropout in {time.time() - t0:.1f}s.")

    # 3. Bayesian Monte Carlo Dropout Epistemic Uncertainty Sampling
    print("\n[4/5] Executing Monte Carlo Dropout Epistemic Uncertainty Estimation (T=10 passes)...")
    sample_t1, sample_t2, sample_mask = next(iter(test_loader))
    t1_chip = sample_t1[0:1].to(device)
    t2_chip = sample_t2[0:1].to(device)
    gt_chip = sample_mask[0, 0].numpy().astype(int)

    mean_prob, std_prob, entropy = estimate_epistemic_uncertainty(
        model=model,
        t1_tensor=t1_chip,
        t2_tensor=t2_chip,
        num_passes=10,
        device=device
    )

    high_uncertainty_mask = (std_prob >= 0.15).astype(int)
    print(f"  • Bayesian Mean Probability (min: {mean_prob.min():.3f}, max: {mean_prob.max():.3f})")
    print(f"  • Epistemic Uncertainty σ   (min: {std_prob.min():.3f}, max: {std_prob.max():.3f}, mean: {std_prob.mean():.3f})")
    print(f"  • Pixels flagged with high epistemic ambiguity (σ >= 0.15): {np.sum(high_uncertainty_mask):,} pixels ({np.mean(high_uncertainty_mask)*100:.1f}%)")

    # 4. Stress Testing: Sensor Noise & Cloud Haze Perturbations
    print("\n[5/5] Running Robustness Stress Tests & Expected Calibration Error (ECE)...")
    stress_tester = RobustnessStressTester(device=device)

    # 4.1 Gaussian Noise Stress Test
    df_noise = stress_tester.run_gaussian_stress_test(
        model=model,
        data_loader=test_loader,
        sigma_levels=[0.0, 0.05, 0.10, 0.15, 0.20]
    )

    # 4.2 Cloud Haze Stress Test
    df_haze = stress_tester.run_cloud_haze_stress_test(
        model=model,
        data_loader=test_loader,
        haze_levels=[0.0, 0.10, 0.20, 0.30]
    )

    # 4.3 Calibration Curve & ECE
    ece_score, df_calib = compute_calibration_curve(
        model=model,
        data_loader=test_loader,
        num_bins=10,
        device=device
    )

    print("\n" + "=" * 80)
    print("📊 ROBUSTNESS STRESS TEST: GAUSSIAN SENSOR NOISE DECAY")
    print("=" * 80)
    print(df_noise.to_string(index=False))
    print("=" * 80)

    print("\n" + "=" * 80)
    print(f"🎯 MODEL CALIBRATION: EXPECTED CALIBRATION ERROR (ECE) = {ece_score*100:.2f}%")
    print("=" * 80)

    out_dir = "outputs/module_18"
    os.makedirs(out_dir, exist_ok=True)

    report_data = {
        "expected_calibration_error": round(ece_score, 4),
        "calibration_bins": df_calib.to_dict(orient="records"),
        "gaussian_noise_degradation": df_noise.to_dict(orient="records"),
        "cloud_haze_degradation": df_haze.to_dict(orient="records"),
        "mean_epistemic_uncertainty": round(float(std_prob.mean()), 4),
        "high_uncertainty_pixels_pct": round(float(np.mean(high_uncertainty_mask) * 100.0), 2)
    }
    with open(os.path.join(out_dir, "robustness_and_uncertainty_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # 5. Generate Visualizations
    print(f"\nGenerating Visualizations in '{out_dir}/'...")

    # --------------------------------------------------------------------------
    # Figure 1: Bayesian Monte Carlo Uncertainty Maps
    # --------------------------------------------------------------------------
    rgb_t1 = make_rgb(sample_t1[0])
    rgb_t2 = make_rgb(sample_t2[0])

    fig1, axes1 = plt.subplots(1, 6, figsize=(22, 4.2), constrained_layout=True)

    axes1[0].imshow(rgb_t1)
    axes1[0].set_title("Pre-Disturbance (T1)\nRGB Composite", fontsize=10, fontweight="bold")
    axes1[0].axis("off")

    axes1[1].imshow(rgb_t2)
    axes1[1].set_title("Post-Disturbance (T2)\nRGB Composite", fontsize=10, fontweight="bold")
    axes1[1].axis("off")

    im_mu = axes1[2].imshow(mean_prob, cmap="inferno", vmin=0, vmax=1)
    axes1[2].set_title("Bayesian Mean Prediction (μ)\n(10 Stochastic MC Passes)", fontsize=10, fontweight="bold")
    axes1[2].axis("off")
    plt.colorbar(im_mu, ax=axes1[2], fraction=0.046, pad=0.04)

    im_sig = axes1[3].imshow(std_prob, cmap="magma", vmin=0, vmax=0.3)
    axes1[3].set_title("Epistemic Uncertainty (σ)\n(Predictive Std Dev)", fontsize=10, fontweight="bold")
    axes1[3].axis("off")
    plt.colorbar(im_sig, ax=axes1[3], fraction=0.046, pad=0.04)

    axes1[4].imshow(high_uncertainty_mask, cmap="Reds", vmin=0, vmax=1)
    axes1[4].set_title("Ambiguous Boundary Mask\n(High Uncertainty σ ≥ 0.15)", fontsize=10, fontweight="bold")
    axes1[4].axis("off")

    axes1[5].imshow(gt_chip, cmap="Greens", vmin=0, vmax=1)
    axes1[5].set_title("Ground Truth Mask\n(True Deforestation)", fontsize=10, fontweight="bold")
    axes1[5].axis("off")

    fig1_path = os.path.join(out_dir, "01_mc_dropout_uncertainty_maps.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  📸 Saved MC Dropout uncertainty maps -> '{fig1_path}'")

    # --------------------------------------------------------------------------
    # Figure 2: Robustness Degradation Curves (Noise & Haze)
    # --------------------------------------------------------------------------
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    # Panel 1: Gaussian Noise Decay
    ax2_1.plot(df_noise["Noise Level (Sigma)"], df_noise["Mean IoU (%)"], "o-", color="#d32f2f", label="Mean IoU (%)", linewidth=2.2)
    ax2_1.plot(df_noise["Noise Level (Sigma)"], df_noise["Dice Score (%)"], "s--", color="#0288d1", label="Dice Score (%)", linewidth=2.2)
    ax2_1.plot(df_noise["Noise Level (Sigma)"], df_noise["Pixel Accuracy (%)"], "^:", color="#2e7d32", label="Pixel Accuracy (%)", linewidth=2.2)
    ax2_1.set_xlabel("Gaussian Sensor Noise Level (σ)", fontweight="bold")
    ax2_1.set_ylabel("Metric Score (%)", fontweight="bold")
    ax2_1.set_title("Model Resilience to Sensor Thermal Noise", fontweight="bold")
    ax2_1.set_ylim(0, 105)
    ax2_1.grid(True, linestyle="--", alpha=0.5)
    ax2_1.legend(loc="lower left")

    # Panel 2: Cloud Haze Decay
    ax2_2.plot(df_haze["Haze Intensity"], df_haze["Mean IoU (%)"], "o-", color="#f57c00", label="Mean IoU (%)", linewidth=2.2)
    ax2_2.plot(df_haze["Haze Intensity"], df_haze["Dice Score (%)"], "s--", color="#7b1fa2", label="Dice Score (%)", linewidth=2.2)
    ax2_2.plot(df_haze["Haze Intensity"], df_haze["Pixel Accuracy (%)"], "^:", color="#2e7d32", label="Pixel Accuracy (%)", linewidth=2.2)
    ax2_2.set_xlabel("Aerosol / Cloud Haze Intensity", fontweight="bold")
    ax2_2.set_ylabel("Metric Score (%)", fontweight="bold")
    ax2_2.set_title("Model Resilience to Atmospheric Haze", fontweight="bold")
    ax2_2.set_ylim(0, 105)
    ax2_2.grid(True, linestyle="--", alpha=0.5)
    ax2_2.legend(loc="lower left")

    fig2_path = os.path.join(out_dir, "02_robustness_stress_test_curves.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  📸 Saved robustness degradation curves -> '{fig2_path}'")

    # --------------------------------------------------------------------------
    # Figure 3: Calibration Curve & Reliability Diagram
    # --------------------------------------------------------------------------
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    # Panel 1: Reliability Diagram
    ax3_1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")
    ax3_1.plot(df_calib["Confidence"], df_calib["Accuracy"], "o-", color="#1976d2", label=f"Model Calibration (ECE = {ece_score*100:.2f}%)", linewidth=2.2)
    ax3_1.set_xlabel("Mean Predicted Confidence", fontweight="bold")
    ax3_1.set_ylabel("Empirical Accuracy", fontweight="bold")
    ax3_1.set_title(f"Reliability Diagram (ECE = {ece_score*100:.2f}%)", fontweight="bold")
    ax3_1.set_xlim(0, 1)
    ax3_1.set_ylim(0, 1.05)
    ax3_1.grid(True, linestyle="--", alpha=0.5)
    ax3_1.legend(loc="upper left")

    # Panel 2: Sample Distribution across Confidence Bins
    ax3_2.bar(df_calib["Bin Center"], df_calib["Count"], width=0.08, color="#0288d1", edgecolor="black", alpha=0.85)
    ax3_2.set_xlabel("Predicted Confidence Bin", fontweight="bold")
    ax3_2.set_ylabel("Number of Pixels", fontweight="bold")
    ax3_2.set_title("Confidence Probability Distribution", fontweight="bold")
    ax3_2.grid(axis="y", linestyle="--", alpha=0.5)

    fig3_path = os.path.join(out_dir, "03_calibration_reliability_diagram.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  📸 Saved calibration reliability diagram -> '{fig3_path}'")

    print("\n" + "=" * 80)
    print("✅ MODULE 18: UNCERTAINTY & ROBUSTNESS COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
