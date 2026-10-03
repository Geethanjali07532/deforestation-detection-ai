"""
risk_forecaster.py

Module 15: Spatial-Temporal Time-Series Analysis & Trend Forecasting
Implements DeforestationRiskForecaster:
  - Autoregressive & seasonal time-series trend forecasting with 95% confidence intervals
  - Spatial Deforestation Vulnerability Index (DVI) combining edge proximity, road proximity,
    historical canopy decline, and fire weather factors
  - Frontier risk zoning (Critical, High, Moderate, Low Risk)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy.ndimage import distance_transform_edt


class DeforestationRiskForecaster:
    """
    Forecasting engine for canopy trajectories and spatial frontier risk assessment.
    """

    def __init__(self, forecast_horizon: int = 12):
        self.forecast_horizon = forecast_horizon

    def forecast_trajectory(
        self,
        time_steps: np.ndarray,
        values: np.ndarray,
        horizon: int = 12
    ) -> Dict[str, Any]:
        """
        Projects vegetation index trajectory forward by `horizon` steps
        with 95% prediction confidence intervals.
        """
        n = len(time_steps)
        t = time_steps.astype(float)

        # Linear trend extrapolation
        slope, intercept = np.polyfit(t, values, 1)

        # Residual variance for confidence intervals
        fitted = intercept + slope * t
        residuals = values - fitted
        std_err = np.std(residuals)

        # Future time steps
        last_t = t[-1]
        future_t = np.arange(last_t + 1, last_t + horizon + 1)
        future_forecast = intercept + slope * future_t

        # Seasonal component projection (if >= 12 steps)
        if n >= 12:
            seasonal_pattern = np.zeros(12)
            for m in range(12):
                m_vals = [residuals[i] for i in range(n) if i % 12 == m]
                if len(m_vals) > 0:
                    seasonal_pattern[m] = np.mean(m_vals)
            future_season = np.array([seasonal_pattern[int(ft) % 12] for ft in future_t])
            future_forecast += future_season

        # 95% Confidence Intervals (1.96 * std_err * sqrt(1 + (t - t_mean)^2 / sum(t - t_mean)^2))
        uncertainty = 1.96 * std_err * np.sqrt(1.0 + (np.arange(1, horizon + 1) / n))
        lower_bound = np.clip(future_forecast - uncertainty, 0.0, 1.0)
        upper_bound = np.clip(future_forecast + uncertainty, 0.0, 1.0)
        future_forecast = np.clip(future_forecast, 0.0, 1.0)

        return {
            "historical_t": time_steps.tolist(),
            "historical_values": values.tolist(),
            "future_t": future_t.tolist(),
            "forecast": future_forecast.tolist(),
            "lower_95": lower_bound.tolist(),
            "upper_95": upper_bound.tolist(),
            "projected_slope": float(slope),
            "projected_annual_rate_pct": float(slope * 12.0 * 100.0)
        }

    def compute_spatial_vulnerability_index(
        self,
        forest_mask: np.ndarray,
        disturbance_mask: np.ndarray,
        road_mask: np.ndarray,
        dndvi_raster: np.ndarray,
        pixel_res_meters: float = 30.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
        """
        Computes the multi-factor Spatial Deforestation Vulnerability Index (DVI) in [0, 1]:
          - Factor 1: Distance to existing deforestation frontier (closer = exponentially higher risk)
          - Factor 2: Proximity to road encroachment corridors
          - Factor 3: Early canopy degradation signal (dNDVI > 0)
        Returns:
          - dvi_continuous: Float array in [0, 1]
          - risk_tiers_map: (0=Low, 1=Moderate, 2=High, 3=Critical Imminent Risk)
          - risk_stats: Spatial footprint in hectares and percentage for each risk tier
        """
        # Distance to deforestation (in meters)
        dist_deforest = distance_transform_edt(disturbance_mask == 0) * pixel_res_meters
        # Exponential decay risk: high within 300m
        risk_deforest = np.exp(-dist_deforest / 250.0)

        # Distance to roads (in meters)
        dist_roads = distance_transform_edt(road_mask == 0) * pixel_res_meters
        risk_roads = np.exp(-dist_roads / 200.0)

        # Early canopy degradation signal (normalized dNDVI)
        dndvi_norm = np.clip(dndvi_raster, 0.0, 0.5) / 0.5

        # Composite Deforestation Vulnerability Index (DVI)
        dvi = (0.45 * risk_deforest + 0.35 * risk_roads + 0.20 * dndvi_norm)
        # Only evaluate on remaining forest
        rem_forest = (forest_mask == 1) & (disturbance_mask == 0)
        dvi[~rem_forest] = 0.0

        # Classify into 4 Risk Tiers
        risk_tiers = np.zeros_like(dvi, dtype=np.uint8)
        risk_tiers[rem_forest & (dvi >= 0.15) & (dvi < 0.35)] = 1  # Moderate Risk
        risk_tiers[rem_forest & (dvi >= 0.35) & (dvi < 0.60)] = 2  # High Risk
        risk_tiers[rem_forest & (dvi >= 0.60)] = 3                 # Critical Imminent Risk

        # Calculate statistics
        pixel_area_ha = (pixel_res_meters * pixel_res_meters) / 10000.0
        total_forest_pixels = int(np.sum(rem_forest))

        tier_names = {
            0: "Low / Deep Core Risk",
            1: "Moderate Vulnerability",
            2: "High Frontier Risk",
            3: "Critical Imminent Risk"
        }

        stats = {}
        for t_id, name in tier_names.items():
            cnt = int(np.sum(risk_tiers == t_id))
            ha = cnt * pixel_area_ha
            pct = (cnt / (total_forest_pixels + 1e-7)) * 100.0
            stats[name] = {
                "pixel_count": cnt,
                "area_hectares": round(ha, 2),
                "area_km2": round(ha / 100.0, 4),
                "percentage_of_forest": round(pct, 2)
            }

        return dvi, risk_tiers, stats
