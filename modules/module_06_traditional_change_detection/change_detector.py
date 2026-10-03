"""
change_detector.py

Module 6: Traditional Forest Change Detection
Implements:
  1. Image Differencing (Euclidean Spectral Vector Distance & Band Differencing)
  2. NDVI Differencing: Delta NDVI = NDVI_before - NDVI_after
  3. Dynamic & Empirical Threshold-based Change Detection
  4. Morphological Noise & False-Positive Filtering
  5. Ground-Truth Quantitative Evaluation (Precision, Recall, F1, IoU, FPR, FNR)
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class TraditionalChangeDetector:
    """Baseline non-deep-learning change detection methods using spectral differencing."""

    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def compute_ndvi(self, bands: np.ndarray) -> np.ndarray:
        """Calculates NDVI from 5-band array [0:B, 1:G, 2:R, 3:NIR, 4:SWIR]."""
        red = bands[2].astype(np.float32)
        nir = bands[3].astype(np.float32)
        ndvi = (nir - red) / (nir + red + self.eps)
        return np.clip(ndvi, -1.0, 1.0)

    def compute_ndvi_difference(self, before_bands: np.ndarray, after_bands: np.ndarray) -> np.ndarray:
        """
        NDVI Differencing:
          Delta NDVI = NDVI_before - NDVI_after
        A large positive difference indicates severe loss of photosynthetic canopy (Deforestation).
        """
        ndvi_before = self.compute_ndvi(before_bands)
        ndvi_after = self.compute_ndvi(after_bands)
        delta_ndvi = ndvi_before - ndvi_after
        return delta_ndvi

    def compute_spectral_vector_difference(self, before_bands: np.ndarray, after_bands: np.ndarray) -> np.ndarray:
        """
        Euclidean Spectral Distance across all 5 bands:
          Distance = sqrt(sum((B_after - B_before)^2))
        """
        diff = after_bands.astype(np.float32) - before_bands.astype(np.float32)
        euclidean_dist = np.sqrt(np.sum(diff**2, axis=0))
        return euclidean_dist

    def detect_change(
        self,
        diff_map: np.ndarray,
        threshold: float = 0.25,
        apply_morph_cleanup: bool = True
    ) -> np.ndarray:
        """
        Generates binary deforestation mask:
          1: Deforested (diff_map >= threshold)
          0: Unchanged (diff_map < threshold)
        Optionally applies morphological opening & closing to eliminate isolated false positives.
        """
        raw_mask = (diff_map >= threshold).astype(np.uint8)

        if apply_morph_cleanup:
            # 3x3 structuring element
            struct = generate_binary_structure(2, 1)
            # Opening removes isolated single-pixel false positives
            cleaned = binary_opening(raw_mask, structure=struct)
            # Closing bridges small gaps inside clearing patches
            cleaned = binary_closing(cleaned, structure=struct)
            return cleaned.astype(np.uint8)

        return raw_mask

    def find_optimal_threshold(
        self,
        diff_map: np.ndarray,
        ground_truth: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Tuple[float, float, Dict[str, List[float]]]:
        """
        Sweeps through candidate thresholds to discover the optimal decision boundary
        maximizing the F1-Score (Dice Score).
        """
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.65, 31)

        f1_scores = []
        ious = []
        precisions = []
        recalls = []

        best_f1 = -1.0
        best_thresh = float(thresholds[0])

        for t in thresholds:
            pred = self.detect_change(diff_map, threshold=float(t), apply_morph_cleanup=True)
            metrics = ChangeEvaluator.evaluate(pred, ground_truth)
            f1_scores.append(metrics["f1_score"])
            ious.append(metrics["iou"])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_thresh = float(t)

        history = {
            "thresholds": list(thresholds),
            "f1_scores": f1_scores,
            "ious": ious,
            "precisions": precisions,
            "recalls": recalls
        }
        return best_thresh, best_f1, history


class ChangeEvaluator:
    """Calculates standardized binary classification & segmentation metrics."""

    @staticmethod
    def evaluate(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """
        Computes TP, FP, TN, FN, Precision, Recall, F1, IoU, Accuracy, FPR, FNR.
        """
        pred = (pred_mask == 1)
        gt = (gt_mask == 1)

        tp = int(np.sum(pred & gt))
        fp = int(np.sum(pred & (~gt)))
        tn = int(np.sum((~pred) & (~gt)))
        fn = int(np.sum((~pred) & gt))

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / (total + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        fnr = fn / (tp + fn + 1e-8)

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "iou": float(iou),
            "fpr": float(fpr),
            "fnr": float(fnr)
        }

    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Traditional Change Detection Evaluation"):
        """Prints formatted metrics report to console."""
        print("=" * 65)
        print(f"📊 {title.upper()}")
        print("=" * 65)
        print(f"  • True Positives (TP)     : {metrics['tp']:>7} pixels")
        print(f"  • False Positives (FP)    : {metrics['fp']:>7} pixels (Over-detection)")
        print(f"  • True Negatives (TN)     : {metrics['tn']:>7} pixels")
        print(f"  • False Negatives (FN)    : {metrics['fn']:>7} pixels (Missed deforestation)")
        print("-" * 65)
        print(f"  • Precision               : {metrics['precision'] * 100:>6.2f}%")
        print(f"  • Recall (Sensitivity)    : {metrics['recall'] * 100:>6.2f}%")
        print(f"  • F1-Score (Dice Score)   : {metrics['f1_score'] * 100:>6.2f}%")
        print(f"  • IoU (Intersection/Union): {metrics['iou'] * 100:>6.2f}%")
        print(f"  • Pixel Accuracy          : {metrics['accuracy'] * 100:>6.2f}%")
        print(f"  • False Positive Rate     : {metrics['fpr'] * 100:>6.2f}%")
        print(f"  • False Negative Rate     : {metrics['fnr'] * 100:>6.2f}%")
        print("=" * 65)
