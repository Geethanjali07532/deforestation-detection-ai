"""
trajectory_modeler.py

Module 15: Spatial-Temporal Time-Series Analysis & Trend Forecasting
Implements VegetationTrajectoryModeler:
  - Harmonic seasonal decomposition: Y(t) = Trend(t) + Seasonality(t) + Residual(t)
  - Abrupt structural breakpoint detection inspired by LandTrendr (Kennedy et al., 2010)
    and BFAST (Verbesselt et al., 2010)
  - Distinguishes permanent deforestation vs. seasonal phenology vs. post-fire recovery
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any


class VegetationTrajectoryModeler:
    """
    Models multi-temporal vegetation trajectories and identifies structural breakpoints.
    """

    def __init__(self, period: int = 12, harmonic_order: int = 2):
        self.period = period  # 12 months for annual cycle
        self.harmonic_order = harmonic_order

    def fit_harmonic_seasonality(self, time_steps: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decomposes time series into: Trend, Seasonality, and Residuals.
        Uses Ordinary Least Squares with harmonic sines and cosines.
        """
        n = len(time_steps)
        t = time_steps.astype(float)

        # Build design matrix: [1, t, cos(2pi*t/P), sin(2pi*t/P), ...]
        cols = [np.ones(n), t]
        for k in range(1, self.harmonic_order + 1):
            cols.append(np.cos(2.0 * np.pi * k * t / self.period))
            cols.append(np.sin(2.0 * np.pi * k * t / self.period))

        X = np.column_stack(cols)
        # Solve least squares: beta = (X^T X)^-1 X^T y
        coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)

        # Trend component: a + b*t
        trend = coeffs[0] + coeffs[1] * t

        # Seasonality component: sum of harmonics
        seasonality = np.zeros(n)
        col_idx = 2
        for k in range(1, self.harmonic_order + 1):
            seasonality += coeffs[col_idx] * np.cos(2.0 * np.pi * k * t / self.period)
            seasonality += coeffs[col_idx + 1] * np.sin(2.0 * np.pi * k * t / self.period)
            col_idx += 2

        residuals = values - (trend + seasonality)
        return trend, seasonality, residuals

    def detect_breakpoints(
        self,
        values: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
        min_distance: int = 4,
        drop_threshold: float = 0.20
    ) -> List[Dict[str, Any]]:
        """
        Detects structural disturbance breakpoints where vegetation index collapses abruptly.
        Returns list of detected break events with drop magnitude and recovery trajectory.
        """
        n = len(values)
        if time_steps is None:
            time_steps = np.arange(n)

        # Compute moving differences and cumulative sum of residuals
        diffs = np.diff(values)
        breakpoints = []

        for i in range(min_distance, n - min_distance):
            pre_window = values[max(0, i - min_distance):i]
            post_window = values[i:min(n, i + min_distance)]

            pre_mean = np.mean(pre_window)
            post_mean = np.mean(post_window)
            magnitude = pre_mean - post_mean

            if magnitude >= drop_threshold:
                # Calculate pre- and post-break slopes
                t_pre = time_steps[max(0, i - min_distance):i]
                t_post = time_steps[i:min(n, i + min_distance)]

                slope_pre = np.polyfit(t_pre, pre_window, 1)[0] if len(t_pre) > 1 else 0.0
                slope_post = np.polyfit(t_post, post_window, 1)[0] if len(t_post) > 1 else 0.0

                # Determine trajectory regime
                if slope_post > 0.015:
                    regime = "Post-Disturbance Regrowth / Secondary Succession"
                elif slope_post < -0.01:
                    regime = "Ongoing Progressive Degradation"
                else:
                    regime = "Permanent Clearcut / Non-Forest Conversion"

                breakpoints.append({
                    "break_index": int(i),
                    "break_time": float(time_steps[i]),
                    "pre_disturbance_mean": round(float(pre_mean), 3),
                    "post_disturbance_mean": round(float(post_mean), 3),
                    "drop_magnitude": round(float(magnitude), 3),
                    "pre_slope": round(float(slope_pre), 4),
                    "recovery_slope": round(float(slope_post), 4),
                    "regime": regime
                })

        # Deduplicate consecutive trigger points (keep maximum drop)
        if len(breakpoints) > 1:
            best_break = max(breakpoints, key=lambda b: b["drop_magnitude"])
            breakpoints = [best_break]

        return breakpoints
