"""
robustness_stress_tester.py

Module 18: Model Evaluation, Robustness & Uncertainty Quantification
Injects synthetic sensor and atmospheric perturbations to evaluate model degradation:
  - Gaussian sensor noise stress testing
  - Atmospheric cloud haze / aerosol scattering
  - Expected Calibration Error (ECE) and reliability diagrams (Naeini et al., 2015)
"""

import os
import sys
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class RobustnessStressTester:
    """
    Stress-testing engine assessing model resilience against real-world
    atmospheric and sensor noise.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    @staticmethod
    def inject_gaussian_noise(tensor: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """Injects additive zero-mean Gaussian sensor noise."""
        if sigma <= 0:
            return tensor
        noise = torch.randn_like(tensor) * sigma
        return torch.clamp(tensor + noise, 0.0, 1.0)

    @staticmethod
    def inject_cloud_haze(tensor: torch.Tensor, haze_intensity: float = 0.2) -> torch.Tensor:
        """
        Simulates atmospheric aerosol scattering:
        Blends satellite scene with additive white scattering and reduces contrast.
        """
        if haze_intensity <= 0:
            return tensor
        hazed = tensor * (1.0 - haze_intensity) + haze_intensity * 0.85
        return torch.clamp(hazed, 0.0, 1.0)

    def evaluate_model_on_loader(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        noise_fn=None
    ) -> Dict[str, float]:
        """Evaluates model with optional online perturbation function applied to T2."""
        model.eval()
        model.to(self.device)

        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

        with torch.no_grad():
            for t1_b, t2_b, mask_b in data_loader:
                t1_b = t1_b.to(self.device)
                if noise_fn is not None:
                    t2_b = noise_fn(t2_b)
                t2_b = t2_b.to(self.device)

                logits = model(t1_b, t2_b)
                preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int).flatten()
                gt = mask_b.numpy().astype(int).flatten()

                total_tp += int(np.sum((preds == 1) & (gt == 1)))
                total_fp += int(np.sum((preds == 1) & (gt == 0)))
                total_tn += int(np.sum((preds == 0) & (gt == 0)))
                total_fn += int(np.sum((preds == 0) & (gt == 1)))

        acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)
        prec = total_tp / (total_tp + total_fp + 1e-8)
        rec = total_tp / (total_tp + total_fn + 1e-8)
        dice = (2 * prec * rec) / (prec + rec + 1e-8)
        iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)

        return {
            "iou": float(iou),
            "dice": float(dice),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec)
        }

    def run_gaussian_stress_test(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        sigma_levels: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20]
    ) -> pd.DataFrame:
        """Evaluates model performance degradation under escalating sensor noise."""
        records = []
        for s in sigma_levels:
            res = self.evaluate_model_on_loader(
                model=model,
                data_loader=data_loader,
                noise_fn=lambda x: self.inject_gaussian_noise(x, sigma=s)
            )
            records.append({
                "Noise Level (Sigma)": s,
                "Mean IoU (%)": round(res["iou"] * 100.0, 2),
                "Dice Score (%)": round(res["dice"] * 100.0, 2),
                "Pixel Accuracy (%)": round(res["accuracy"] * 100.0, 2),
                "Precision (%)": round(res["precision"] * 100.0, 2),
                "Recall (%)": round(res["recall"] * 100.0, 2)
            })
        return pd.DataFrame(records)

    def run_cloud_haze_stress_test(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        haze_levels: List[float] = [0.0, 0.10, 0.20, 0.30]
    ) -> pd.DataFrame:
        """Evaluates model performance degradation under escalating atmospheric haze."""
        records = []
        for h in haze_levels:
            res = self.evaluate_model_on_loader(
                model=model,
                data_loader=data_loader,
                noise_fn=lambda x: self.inject_cloud_haze(x, haze_intensity=h)
            )
            records.append({
                "Haze Intensity": h,
                "Mean IoU (%)": round(res["iou"] * 100.0, 2),
                "Dice Score (%)": round(res["dice"] * 100.0, 2),
                "Pixel Accuracy (%)": round(res["accuracy"] * 100.0, 2),
                "Precision (%)": round(res["precision"] * 100.0, 2),
                "Recall (%)": round(res["recall"] * 100.0, 2)
            })
        return pd.DataFrame(records)


def compute_calibration_curve(
    model: nn.Module,
    data_loader: DataLoader,
    num_bins: int = 10,
    device: str = "cpu"
) -> Tuple[float, pd.DataFrame]:
    """
    Computes Expected Calibration Error (ECE) and binned accuracy vs confidence.
    """
    model.eval()
    model.to(device)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for t1_b, t2_b, mask_b in data_loader:
            t1_b, t2_b = t1_b.to(device), t2_b.to(device)
            logits = model(t1_b, t2_b)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = mask_b.numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(targets)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_records = []
    ece = 0.0
    total_samples = len(all_probs)

    for i in range(num_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        in_bin = (all_probs >= bin_lower) & (all_probs < bin_upper) if i < num_bins - 1 else (all_probs >= bin_lower) & (all_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if np.sum(in_bin) > 0:
            bin_acc = np.mean(all_targets[in_bin] == (all_probs[in_bin] >= 0.5))
            bin_conf = np.mean(all_probs[in_bin])
            bin_error = np.abs(bin_acc - bin_conf)
            ece += (np.sum(in_bin) / total_samples) * bin_error

            bin_records.append({
                "Bin Center": (bin_lower + bin_upper) / 2.0,
                "Accuracy": float(bin_acc),
                "Confidence": float(bin_conf),
                "Count": int(np.sum(in_bin))
            })
        else:
            bin_records.append({
                "Bin Center": (bin_lower + bin_upper) / 2.0,
                "Accuracy": 0.0,
                "Confidence": (bin_lower + bin_upper) / 2.0,
                "Count": 0
            })

    df_calib = pd.DataFrame(bin_records)
    return float(ece), df_calib
