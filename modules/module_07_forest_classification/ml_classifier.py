"""
ml_classifier.py

Module 7: Forest vs Non-Forest Classification
Implements:
  1. PixelFeatureExtractor: Extracts 13 spectral and index features per pixel.
  2. ForestMLClassifierSuite: Trains, benchmarks, and compares:
     - Random Forest
     - Support Vector Machine (SVM)
     - XGBoost
  3. Feature Importance Analysis (Which spectral bands matter most for forest detection).
  4. Spatial Mask Generation & Evaluation.
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, jaccard_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class PixelFeatureExtractor:
    """
    Transforms multi-band satellite rasters into rich tabular feature matrices.
    For each pixel (x, y), extracts 13 biophysical and spectral indicators:
      1. Blue (490 nm)
      2. Green (560 nm)
      3. Red (665 nm)
      4. NIR (842 nm)
      5. SWIR (1610 nm)
      6. NDVI: (NIR - Red) / (NIR + Red)
      7. EVI: Enhanced Vegetation Index
      8. SAVI: Soil Adjusted Vegetation Index
      9. NDWI_Moisture: (NIR - SWIR) / (NIR + SWIR)
      10. NBR: Normalized Burn Ratio
      11. Ratio NIR/Red: Simple Ratio
      12. Ratio Red/Green: Soil/Vegetation color ratio
      13. Ratio SWIR/NIR: Canopy moisture stress ratio
    """

    FEATURE_NAMES = [
        "Blue", "Green", "Red", "NIR", "SWIR",
        "NDVI", "EVI", "SAVI", "NDWI_Moist", "NBR",
        "Ratio_NIR_Red", "Ratio_Red_Green", "Ratio_SWIR_NIR"
    ]

    @classmethod
    def extract_features(cls, bands: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """
        Input: bands of shape (5, H, W)
        Returns: feature array of shape (H*W, 13)
        """
        blue = bands[0].astype(np.float32)
        green = bands[1].astype(np.float32)
        red = bands[2].astype(np.float32)
        nir = bands[3].astype(np.float32)
        swir = bands[4].astype(np.float32)

        # Spectral Indices
        ndvi = (nir - red) / (nir + red + eps)
        evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps)
        savi = ((nir - red) / (nir + red + 0.5 + eps)) * 1.5
        ndwi_m = (nir - swir) / (nir + swir + eps)
        nbr = (nir - swir) / (nir + swir + eps)

        # Ratios
        ratio_nir_red = nir / (red + eps)
        ratio_red_green = red / (green + eps)
        ratio_swir_nir = swir / (nir + eps)

        feature_stack = np.stack([
            blue, green, red, nir, swir,
            ndvi, np.clip(evi, -1.0, 1.5), np.clip(savi, -1.0, 1.0),
            ndwi_m, nbr,
            np.clip(ratio_nir_red, 0, 50),
            np.clip(ratio_red_green, 0, 20),
            np.clip(ratio_swir_nir, 0, 20)
        ], axis=0)

        # Reshape to (N, 13)
        num_features = feature_stack.shape[0]
        h, w = bands.shape[1], bands.shape[2]
        return feature_stack.reshape(num_features, h * w).T

    @classmethod
    def extract_training_sample(
        cls,
        bands: np.ndarray,
        forest_labels: np.ndarray,
        sample_size: int = 15000,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts a balanced or stratified subset of pixel features and labels.
        forest_labels: 1 = Forest, 0 = Non-Forest.
        """
        X = cls.extract_features(bands)
        y = forest_labels.flatten().astype(np.int32)

        rng = np.random.default_rng(seed)
        forest_indices = np.where(y == 1)[0]
        nonforest_indices = np.where(y == 0)[0]

        half_sample = sample_size // 2
        f_sub = rng.choice(forest_indices, size=min(half_sample, len(forest_indices)), replace=False)
        nf_sub = rng.choice(nonforest_indices, size=min(half_sample, len(nonforest_indices)), replace=False)

        selected_indices = np.concatenate([f_sub, nf_sub])
        rng.shuffle(selected_indices)

        return X[selected_indices], y[selected_indices]


class ForestMLClassifierSuite:
    """
    Trains, benchmarks, and compares:
      1. Random Forest Classifier
      2. Support Vector Machine (LinearSVC with calibration)
      3. XGBoost Classifier
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

        # Models dictionary
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                n_jobs=-1,
                random_state=random_state
            ),
            "SVM (Linear)": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", CalibratedClassifierCV(
                    LinearSVC(dual=False, random_state=random_state, max_iter=2000),
                    cv=3
                ))
            ]),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1
            )
        }

        self.trained_models = {}
        self.training_times = {}

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains all three models and records execution durations."""
        print(f"Training ML Classifier Suite on {len(X_train)} sampled pixels across 13 features...")
        for name, model in self.models.items():
            t0 = time.time()
            model.fit(X_train, y_train)
            dur = time.time() - t0
            self.trained_models[name] = model
            self.training_times[name] = dur
            print(f"  • {name:<15} trained in {dur:.2f} seconds.")

    def evaluate_model(self, name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluates a single model against test pixels."""
        model = self.trained_models[name]
        t0 = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - t0) * 1000.0  # ms

        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = 0.0

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        iou = jaccard_score(y_test, y_pred, zero_division=0)

        return {
            "model_name": name,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "iou": float(iou),
            "roc_auc": float(auc),
            "inference_time_ms": round(inference_time, 2),
            "train_time_sec": round(self.training_times[name], 2)
        }

    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluates all models and compiles a comparative benchmark leaderboard."""
        records = [self.evaluate_model(name, X_test, y_test) for name in self.trained_models]
        df = pd.DataFrame(records).sort_values(by="f1_score", ascending=False).reset_index(drop=True)
        return df

    def predict_scene_mask(self, name: str, bands: np.ndarray) -> np.ndarray:
        """Runs full 2D scene inference returning a (H, W) Forest/Non-Forest mask."""
        model = self.trained_models[name]
        h, w = bands.shape[1], bands.shape[2]
        X = PixelFeatureExtractor.extract_features(bands)
        y_pred = model.predict(X)
        return y_pred.reshape(h, w).astype(np.uint8)

    def get_feature_importances(self) -> Dict[str, pd.Series]:
        """Extracts relative feature importances for tree-based models (RF & XGBoost)."""
        importances = {}
        feat_names = PixelFeatureExtractor.FEATURE_NAMES

        # Random Forest
        rf = self.trained_models.get("Random Forest")
        if rf is not None and hasattr(rf, "feature_importances_"):
            importances["Random Forest"] = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)

        # XGBoost
        xgb = self.trained_models.get("XGBoost")
        if xgb is not None and hasattr(xgb, "feature_importances_"):
            importances["XGBoost"] = pd.Series(xgb.feature_importances_, index=feat_names).sort_values(ascending=False)

        return importances
