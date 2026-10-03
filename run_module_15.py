"""
run_module_15.py

Master execution script for Module 15: Spatial-Temporal Time-Series Analysis & Trend Forecasting.
Accomplishes:
  1. Multi-temporal trajectory modeling across 4 ecological forest profiles:
     - Stable Intact Forest
     - Abrupt Clearcut Deforestation
     - Wildfire Disturbance with Secondary Regrowth
     - Progressive Canopy Degradation
  2. Harmonic seasonal decomposition: Trend + Seasonality + Residuals
  3. Abrupt structural break detection (LandTrendr & BFAST methodology)
  4. 12-month autoregressive forward trend forecasting with 95% confidence intervals
  5. Spatial Deforestation Vulnerability Index (DVI) & Frontier Risk Mapping
  6. Publication-grade figures in outputs/module_15/
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
from modules.module_14_pattern_analysis.morphological_analyzer import PatchMorphologyExtractor
from modules.module_14_pattern_analysis.driver_classifier import DisturbanceDriverClassifier
from modules.module_15_timeseries_analysis.trajectory_modeler import VegetationTrajectoryModeler
from modules.module_15_timeseries_analysis.risk_forecaster import DeforestationRiskForecaster


def generate_synthetic_36mo_trajectories():
    """Generates realistic 36-month monthly NDVI trajectories for 4 distinct land-cover regimes."""
    np.random.seed(42)
    months = np.arange(1, 37)  # 3 years of monthly satellite passes
    seasonal_cycle = 0.06 * np.sin(2.0 * np.pi * months / 12.0)

    # 1. Stable Forest: Mean 0.82 + seasonal wobble + noise
    stable = 0.82 + seasonal_cycle + np.random.normal(0, 0.02, 36)

    # 2. Abrupt Clearcut: Stable until month 18, then abrupt drop to 0.22 (pasture/bare ground)
    clearcut = 0.80 + seasonal_cycle + np.random.normal(0, 0.02, 36)
    clearcut[18:] = 0.22 + 0.03 * np.sin(2.0 * np.pi * months[18:] / 12.0) + np.random.normal(0, 0.02, 18)

    # 3. Fire Disturbance with Secondary Regrowth: Fire at month 14, followed by positive recovery slope
    fire_regrowth = 0.81 + seasonal_cycle + np.random.normal(0, 0.02, 36)
    fire_regrowth[14:16] = 0.18 + np.random.normal(0, 0.02, 2)
    recovery_time = np.arange(20)
    fire_regrowth[16:] = 0.25 + 0.022 * recovery_time + 0.04 * np.sin(2.0 * np.pi * recovery_time / 12.0) + np.random.normal(0, 0.02, 20)

    # 4. Progressive Canopy Degradation: Gradual persistent decline (selective logging / drought)
    degradation = 0.83 - 0.011 * months + seasonal_cycle + np.random.normal(0, 0.02, 36)

    return {
        "months": months,
        "Stable Intact Forest": np.clip(stable, 0.0, 1.0),
        "Abrupt Clearcut": np.clip(clearcut, 0.0, 1.0),
        "Wildfire with Regrowth": np.clip(fire_regrowth, 0.0, 1.0),
        "Progressive Degradation": np.clip(degradation, 0.0, 1.0)
    }


def main():
    print("\n" + "=" * 80)
    print("🛰️  MODULE 15: SPATIAL-TEMPORAL TIME-SERIES & TREND FORECASTING")
    print("=" * 80)

    # 1. Initialize Modeling Engines
    print("\n[1/5] Initializing Trajectory Modeler & Risk Forecaster...")
    modeler = VegetationTrajectoryModeler(period=12, harmonic_order=2)
    forecaster = DeforestationRiskForecaster(forecast_horizon=12)

    # 2. Time-Series Trajectory Breakpoint Analysis
    print("\n[2/5] Simulating Multi-Year (36-Month) Satellite Trajectories & Break Detection...")
    trajectories = generate_synthetic_36mo_trajectories()
    months = trajectories["months"]

    breaks_results = {}
    print("\n" + "=" * 85)
    print("📊 MULTI-TEMPORAL TRAJECTORY BREAKPOINT DETECTION (LANDTRENDR / BFAST)")
    print("=" * 85)
    print(f"{'Profile':<28} {'Breakpoint':<14} {'Pre-NDVI':<12} {'Post-NDVI':<12} {'Drop':<10} {'Regime':<30}")
    print("-" * 85)

    for profile_name in ["Stable Intact Forest", "Abrupt Clearcut", "Wildfire with Regrowth", "Progressive Degradation"]:
        vals = trajectories[profile_name]
        detected = modeler.detect_breakpoints(vals, time_steps=months, min_distance=4, drop_threshold=0.20)
        breaks_results[profile_name] = detected

        if len(detected) > 0:
            b = detected[0]
            print(f"{profile_name:<28} Month {int(b['break_time']):<8} {b['pre_disturbance_mean']:<12.3f} {b['post_disturbance_mean']:<12.3f} -{b['drop_magnitude']:<9.3f} {b['regime']}")
        else:
            print(f"{profile_name:<28} {'None detected':<14} {'--':<12} {'--':<12} {'--':<10} Stable / Gradual Trend")
    print("=" * 85)

    # 3. Harmonic Decomposition & 12-Month Predictive Forecast
    print("\n[3/5] Performing Harmonic Decomposition & 12-Month Autoregressive Forecast...")
    trend_deg, season_deg, resid_deg = modeler.fit_harmonic_seasonality(months, trajectories["Progressive Degradation"])
    forecast_results = forecaster.forecast_trajectory(months, trajectories["Progressive Degradation"], horizon=12)

    print(f"  • Projected annual canopy loss rate: {forecast_results['projected_annual_rate_pct']:.2f}% per year")
    print(f"  • 12-month projected NDVI index   : {forecast_results['forecast'][-1]:.3f} (95% CI: [{forecast_results['lower_95'][-1]:.3f}, {forecast_results['upper_95'][-1]:.3f}])")

    # 4. Spatial Deforestation Frontier Vulnerability Analysis
    print("\n[4/5] Computing Spatial Deforestation Vulnerability Index (DVI) on Test Scene...")
    with rasterio.open("dataset/test/before/scene_test_001.tif") as s1:
        scene_t1 = s1.read()
    with rasterio.open("dataset/test/after/scene_test_001.tif") as s2:
        scene_t2 = s2.read()
    with rasterio.open("dataset/test/masks/scene_test_001.tif") as sm:
        mask = sm.read(1)

    calc = DisturbanceSeverityCalculator()
    extractor = PatchMorphologyExtractor(min_patch_pixels=4)
    driver_clf = DisturbanceDriverClassifier()

    idx_dict = calc.compute_all_indices(scene_t1, scene_t2)
    patches, _ = extractor.extract_patch_metrics(mask, dnbr_raster=idx_dict["dnbr"])

    # Extract road mask from patches
    road_mask = np.zeros_like(mask, dtype=np.uint8)
    for p in patches:
        if driver_clf.classify_single_patch(p) == "Road Encroachment":
            cnt = p["contour"]
            import cv2
            cv2.drawContours(road_mask, [cnt], -1, 1, thickness=-1)

    # Forest baseline (NDVI >= 0.50 at T1)
    forest_mask = (calc.compute_ndvi(scene_t1[3], scene_t1[2]) >= 0.50).astype(np.uint8)

    dvi_map, risk_tiers_map, risk_stats = forecaster.compute_spatial_vulnerability_index(
        forest_mask=forest_mask,
        disturbance_mask=mask,
        road_mask=road_mask,
        dndvi_raster=idx_dict["dndvi"],
        pixel_res_meters=30.0
    )

    print("\n" + "=" * 85)
    print("🌲 SPATIAL DEFORESTATION FRONTIER VULNERABILITY AUDIT")
    print("=" * 85)
    print(f"{'Vulnerability Tier':<28} {'Pixels':<12} {'Area (ha)':<15} {'Area (km²)':<14} {'Share (%)':<10}")
    print("-" * 85)
    for name, s in risk_stats.items():
        print(f"{name:<28} {s['pixel_count']:<12,} {s['area_hectares']:<15,} {s['area_km2']:<14} {s['percentage_of_forest']:>5.2f}%")
    print("=" * 85)

    out_dir = "outputs/module_15"
    os.makedirs(out_dir, exist_ok=True)

    # Save JSON summary
    summary_json = {
        "breakpoints_detected": breaks_results,
        "forecast_12mo": {
            "projected_slope": forecast_results["projected_slope"],
            "projected_annual_rate_pct": forecast_results["projected_annual_rate_pct"],
            "t_plus_12_ndvi": forecast_results["forecast"][-1],
            "ci_95": [forecast_results["lower_95"][-1], forecast_results["upper_95"][-1]]
        },
        "spatial_frontier_risk_stats": risk_stats
    }
    with open(os.path.join(out_dir, "timeseries_risk_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    # 5. Generate Visualizations
    print(f"\n[5/5] Generating Visualizations in '{out_dir}/'...")

    # --------------------------------------------------------------------------
    # Figure 1: Multi-Temporal Trajectory Curves & Breakpoints
    # --------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    profiles = [
        ("Stable Intact Forest", axes1[0, 0], "#2e7d32"),
        ("Abrupt Clearcut", axes1[0, 1], "#d32f2f"),
        ("Wildfire with Regrowth", axes1[1, 0], "#f57c00"),
        ("Progressive Degradation", axes1[1, 1], "#7b1fa2")
    ]

    for name, ax, col in profiles:
        vals = trajectories[name]
        ax.plot(months, vals, "o-", color=col, label="Observed NDVI", linewidth=2, markersize=4)

        b_list = breaks_results[name]
        if len(b_list) > 0:
            b = b_list[0]
            b_idx = b["break_time"]
            ax.axvline(x=b_idx, color="black", linestyle="--", linewidth=1.8, label=f"Breakpoint (Month {int(b_idx)})")
            ax.scatter([b_idx], [vals[int(b_idx)-1]], s=180, color="red", zorder=5, edgecolors="black", label=f"Drop: -{b['drop_magnitude']:.2f}")
            ax.text(b_idx + 0.5, 0.45, f"Regime:\n{b['regime']}", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (Months)", fontweight="bold")
        ax.set_ylabel("NDVI Vegetation Index", fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=8)

    fig1_path = os.path.join(out_dir, "01_timeseries_trajectory_breakpoints.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  📸 Saved trajectory breakpoint plots -> '{fig1_path}'")

    # --------------------------------------------------------------------------
    # Figure 2: Harmonic Seasonal Decomposition & 12-Month Forecast
    # --------------------------------------------------------------------------
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(18, 5.5), constrained_layout=True)

    # Subplot 1: Harmonic Decomposition
    vals_deg = trajectories["Progressive Degradation"]
    ax2_1.plot(months, vals_deg, "k-", label="Observed NDVI", alpha=0.7)
    ax2_1.plot(months, trend_deg, "r--", linewidth=2.2, label=f"Inter-Annual Trend (Slope: {forecast_results['projected_slope']:.4f}/mo)")
    ax2_1.plot(months, trend_deg + season_deg, "b-", linewidth=1.5, label="Fitted (Trend + Seasonality)")
    ax2_1.set_xlabel("Time (Months)", fontweight="bold")
    ax2_1.set_ylabel("NDVI Index", fontweight="bold")
    ax2_1.set_title("Harmonic Seasonal Decomposition: Y(t) = Trend + Season + Res", fontweight="bold")
    ax2_1.set_ylim(0.3, 1.0)
    ax2_1.grid(True, linestyle="--", alpha=0.5)
    ax2_1.legend(loc="lower left", fontsize=9)

    # Subplot 2: Predictive Forecast with 95% Confidence Interval
    hist_t = forecast_results["historical_t"]
    fut_t = forecast_results["future_t"]
    all_t = hist_t + fut_t

    ax2_2.plot(hist_t, forecast_results["historical_values"], "k-o", label="Historical Observations (36 Mo)", markersize=4)
    ax2_2.plot(fut_t, forecast_results["forecast"], "r--o", label="12-Month Forward Forecast", markersize=4, linewidth=2)
    ax2_2.fill_between(fut_t, forecast_results["lower_95"], forecast_results["upper_95"],
                       color="red", alpha=0.2, label="95% Confidence Interval")
    ax2_2.axvline(x=36, color="gray", linestyle=":", linewidth=1.5)
    ax2_2.text(36.5, 0.92, "Forecast Horizon (12 Months)", fontsize=9, fontweight="bold", color="darkred")

    ax2_2.set_xlabel("Time (Months)", fontweight="bold")
    ax2_2.set_ylabel("NDVI Index", fontweight="bold")
    ax2_2.set_title(f"Canopy Trajectory Projection (Annual Loss: {forecast_results['projected_annual_rate_pct']:.2f}%/yr)", fontweight="bold")
    ax2_2.set_ylim(0.3, 1.0)
    ax2_2.grid(True, linestyle="--", alpha=0.5)
    ax2_2.legend(loc="lower left", fontsize=9)

    fig2_path = os.path.join(out_dir, "02_harmonic_decomposition_and_forecast.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  📸 Saved harmonic decomposition & forecast -> '{fig2_path}'")

    # --------------------------------------------------------------------------
    # Figure 3: Spatial Deforestation Frontier Vulnerability Map
    # --------------------------------------------------------------------------
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)

    # Panel 1: Continuous DVI Heatmap
    im_dvi = ax3_1.imshow(dvi_map, cmap="YlOrRd", vmin=0, vmax=0.8)
    ax3_1.set_title("Deforestation Vulnerability Index (DVI) Continuous Heatmap", fontsize=11, fontweight="bold")
    ax3_1.axis("off")
    cbar1 = plt.colorbar(im_dvi, ax=ax3_1, fraction=0.046, pad=0.04)
    cbar1.set_label("Vulnerability Score [0, 1]", fontweight="bold")

    # Panel 2: 4-Tier Frontier Risk Map
    cmap_risk = mcolors.ListedColormap(["#1b5e20", "#fbc02d", "#f57c00", "#d32f2f"])
    bounds_risk = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_risk = mcolors.BoundaryNorm(bounds_risk, cmap_risk.N)

    im_risk = ax3_2.imshow(risk_tiers_map, cmap=cmap_risk, norm=norm_risk)
    ax3_2.set_title("Categorized Forest Frontier Risk Zoning Map", fontsize=11, fontweight="bold")
    ax3_2.axis("off")
    cbar2 = plt.colorbar(im_risk, ax=ax3_2, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar2.ax.set_yticklabels([
        f"0: Core Forest ({risk_stats['Low / Deep Core Risk']['percentage_of_forest']:.1f}%)",
        f"1: Moderate ({risk_stats['Moderate Vulnerability']['percentage_of_forest']:.1f}%)",
        f"2: High Risk ({risk_stats['High Frontier Risk']['percentage_of_forest']:.1f}%)",
        f"3: Critical ({risk_stats['Critical Imminent Risk']['percentage_of_forest']:.1f}%)"
    ], fontsize=8.5, fontweight="bold")

    fig3_path = os.path.join(out_dir, "03_spatial_deforestation_frontier_risk_map.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  📸 Saved spatial vulnerability risk map -> '{fig3_path}'")

    print("\n" + "=" * 80)
    print("✅ MODULE 15: TIME-SERIES & RISK FORECASTING COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
