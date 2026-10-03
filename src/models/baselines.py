"""
src/models/baselines.py
Classical Baselines: Spectral Differencing (dNDVI) and Machine Learning Pixel Classifiers.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score

from src.indices import extract_pixel_features_for_ml, compute_baseline_change_mask


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes IoU, Dice, Precision, Recall, and Accuracy."""
    yt = y_true.flatten().astype(int)
    yp = y_pred.flatten().astype(int)

    iou = float(jaccard_score(yt, yp, average="binary", zero_division=0))
    dice = float(f1_score(yt, yp, average="binary", zero_division=0))
    prec = float(precision_score(yt, yp, average="binary", zero_division=0))
    rec = float(recall_score(yt, yp, average="binary", zero_division=0))
    acc = float(accuracy_score(yt, yp))

    # Confusion matrix elements for false positive/negative rates
    fp = np.sum((yt == 0) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    tn = np.sum((yt == 0) & (yp == 0))
    tp = np.sum((yt == 1) & (yp == 1))

    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))

    return {
        "iou": round(iou * 100.0, 2),
        "dice": round(dice * 100.0, 2),
        "precision": round(prec * 100.0, 2),
        "recall": round(rec * 100.0, 2),
        "pixel_accuracy": round(acc * 100.0, 2),
        "false_positive_rate": round(fpr * 100.0, 2),
        "false_negative_rate": round(fnr * 100.0, 2)
    }


class NDVIDifferencingBaseline:
    """Zero-parameter spectral differencing baseline on dNDVI."""

    def __init__(self, threshold: float = 0.30):
        self.threshold = threshold

    def predict(self, dndvi: np.ndarray) -> np.ndarray:
        return (dndvi >= self.threshold).astype(np.uint8)

    def evaluate(self, dndvi: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        pred = self.predict(dndvi)
        return compute_binary_metrics(ground_truth, pred)


class RandomForestBaseline:
    """Random Forest Classifier on extracted multispectral pixel features."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 12, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_raster(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        c, h, w = t1.shape
        X = extract_pixel_features_for_ml(t1, t2)
        preds = self.predict(X)
        return preds.reshape(h, w).astype(np.uint8)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        self.model = joblib.load(filepath)
        self.is_fitted = True


class HistGradientBoostingBaseline:
    """Fast Histogram-based Gradient Boosting Classifier on pixel features."""

    def __init__(self, max_iter: int = 100, max_depth: int = 8, random_state: int = 42):
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_raster(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        c, h, w = t1.shape
        X = extract_pixel_features_for_ml(t1, t2)
        preds = self.predict(X)
        return preds.reshape(h, w).astype(np.uint8)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        self.model = joblib.load(filepath)
        self.is_fitted = True
