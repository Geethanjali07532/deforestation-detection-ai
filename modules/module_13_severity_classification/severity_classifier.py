"""
severity_classifier.py

Module 13: Deforestation Severity & Post-Disturbance Classification
Machine learning multi-class classifier predicting forest disturbance severity tiers:
  - Extracts 18 multi-temporal spectral and index features per pixel
  - Trains an optimized Random Forest / Gradient Boosted classifier
  - Evaluates multi-class confusion matrix, precision, recall, and macro-F1
"""

import os
import sys
import time
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from .severity_indices import DisturbanceSeverityCalculator

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ForestDisturbanceClassifier:
    """
    Multi-class classifier predicting 4 disturbance severity tiers:
      Tier 0: Undisturbed / Stable Forest
      Tier 1: Low Severity (Selective Logging)
      Tier 2: Moderate Severity (Heavy Thinning)
      Tier 3: High Severity (Stand-Replacing Fire / Clearcut)
    """

    FEATURE_NAMES = [
        "blue_t1", "green_t1", "red_t1", "nir_t1", "swir_t1",
        "blue_t2", "green_t2", "red_t2", "nir_t2", "swir_t2",
        "nbr_pre", "nbr_post", "dnbr", "rdnbr", "rbr",
        "dndvi", "dndmi", "nir_ratio"
    ]

    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.calc = DisturbanceSeverityCalculator()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def extract_pixel_features(self, t1_bands: np.ndarray, t2_bands: np.ndarray) -> np.ndarray:
        """
        Extracts 18 multi-temporal features for every pixel in (5, H, W) rasters.
        Returns feature matrix of shape (H*W, 18).
        """
        idx = self.calc.compute_all_indices(t1_bands, t2_bands)

        b1, g1, r1, n1, s1 = [t1_bands[i].flatten() for i in range(5)]
        b2, g2, r2, n2, s2 = [t2_bands[i].flatten() for i in range(5)]

        nbr_pre = idx["nbr_pre"].flatten()
        nbr_post = idx["nbr_post"].flatten()
        dnbr = idx["dnbr"].flatten()
        rdnbr = idx["rdnbr"].flatten()
        rbr = idx["rbr"].flatten()
        dndvi = idx["dndvi"].flatten()
        dndmi = idx["dndmi"].flatten()
        nir_ratio = (n2 / (n1 + 1e-7)).flatten()

        X = np.column_stack([
            b1, g1, r1, n1, s1,
            b2, g2, r2, n2, s2,
            nbr_pre, nbr_post, dnbr, rdnbr, rbr,
            dndvi, dndmi, nir_ratio
        ])
        return X

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the Random Forest severity classifier."""
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - t0
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts discrete severity tiers (0, 1, 2, 3)."""
        return self.model.predict(X)

    def predict_spatial_raster(self, t1_bands: np.ndarray, t2_bands: np.ndarray) -> np.ndarray:
        """Processes full (5, H, W) scenes and returns (H, W) severity tier map."""
        _, h, w = t1_bands.shape
        X = self.extract_pixel_features(t1_bands, t2_bands)
        preds = self.predict(X)
        return preds.reshape(h, w).astype(np.uint8)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Calculates multi-class classification metrics."""
        t0 = time.time()
        y_pred = self.predict(X_test)
        infer_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "confusion_matrix": cm,
            "inference_time_sec": infer_time
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns sorted feature importance table."""
        imp = self.model.feature_importances_
        df = pd.DataFrame({
            "Feature": self.FEATURE_NAMES,
            "Importance": imp
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        return df
