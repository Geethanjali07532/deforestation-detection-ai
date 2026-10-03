"""
src/inference.py
Operational Model Inference Engine & Explainability (Grad-CAM / Feature Activation Maps).
Loads trained model checkpoints, runs inference on multi-spectral satellite rasters,
and generates spatial attention heatmaps.
"""

import os
import joblib
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from src.models.unet import ForestUNet
from src.models.siamese import SiameseChangeDetector
from src.models.early_fusion import EarlyFusion4ClassNet
from src.indices import extract_pixel_features_for_ml


class ModelInferenceEngine:
    """
    Central inference engine managing real model checkpoints and activations.
    """
    def __init__(self, models_dir: str = "models", device: Optional[str] = None):
        self.models_dir = models_dir
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.unet_model: Optional[ForestUNet] = None
        self.siamese_model: Optional[SiameseChangeDetector] = None
        self.ef_model: Optional[EarlyFusion4ClassNet] = None
        self.rf_model: Optional[Any] = None

        self._load_checkpoints()

    def _load_checkpoints(self):
        """Loads available weights from disk."""
        unet_path = os.path.join(self.models_dir, "unet_forest.pth")
        if os.path.exists(unet_path):
            self.unet_model = ForestUNet(in_channels=5, num_classes=1, base_features=16).to(self.device)
            self.unet_model.load_state_dict(torch.load(unet_path, map_location=self.device))
            self.unet_model.eval()

        siamese_path = os.path.join(self.models_dir, "siamese_change.pth")
        if os.path.exists(siamese_path):
            self.siamese_model = SiameseChangeDetector(in_channels=5, base_features=16).to(self.device)
            self.siamese_model.load_state_dict(torch.load(siamese_path, map_location=self.device))
            self.siamese_model.eval()

        ef_path = os.path.join(self.models_dir, "early_fusion_4class.pth")
        if os.path.exists(ef_path):
            self.ef_model = EarlyFusion4ClassNet(in_channels=10, num_classes=4, base_features=16).to(self.device)
            self.ef_model.load_state_dict(torch.load(ef_path, map_location=self.device))
            self.ef_model.eval()

        rf_path = os.path.join(self.models_dir, "random_forest_baseline.joblib")
        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)

    def is_ready(self) -> Dict[str, bool]:
        """Reports checkpoint availability."""
        return {
            "unet": self.unet_model is not None,
            "siamese": self.siamese_model is not None,
            "early_fusion": self.ef_model is not None,
            "random_forest": self.rf_model is not None
        }

    def predict_forest_segmentation(self, t1_bands: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        """Runs ForestUNet inference to generate binary forest mask (1=Forest, 0=Non-Forest)."""
        if self.unet_model is None:
            raise FileNotFoundError("ForestUNet checkpoint 'models/unet_forest.pth' not found.")

        tensor_t1 = torch.from_numpy(t1_bands[:5]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.unet_model(tensor_t1)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        return (probs >= threshold).astype(np.uint8)

    def predict_siamese_change(self, t1_bands: np.ndarray, t2_bands: np.ndarray, threshold: float = 0.50) -> Tuple[np.ndarray, np.ndarray]:
        """Runs SiameseChangeDetector inference. Returns (continuous_probability_map, binary_mask)."""
        if self.siamese_model is None:
            raise FileNotFoundError("SiameseChangeDetector checkpoint 'models/siamese_change.pth' not found.")

        t1_t = torch.from_numpy(t1_bands[:5]).unsqueeze(0).to(self.device)
        t2_t = torch.from_numpy(t2_bands[:5]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.siamese_model(t1_t, t2_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        mask = (probs >= threshold).astype(np.uint8)
        return probs, mask

    def predict_early_fusion_4class(self, t1_bands: np.ndarray, t2_bands: np.ndarray) -> np.ndarray:
        """Runs EarlyFusion4ClassNet inference. Returns (H, W) array with classes 0, 1, 2, 3."""
        if self.ef_model is None:
            raise FileNotFoundError("EarlyFusion4ClassNet checkpoint 'models/early_fusion_4class.pth' not found.")

        t1_t = torch.from_numpy(t1_bands[:5]).unsqueeze(0).to(self.device)
        t2_t = torch.from_numpy(t2_bands[:5]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.ef_model(t1_t, t2_t)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_classes = np.argmax(probs, axis=0).astype(np.uint8)

        return pred_classes

    def compute_gradcam_attention(self, t1_bands: np.ndarray, t2_bands: np.ndarray) -> np.ndarray:
        """
        Computes Grad-CAM / feature difference attention heatmap showing which
        spatial regions most strongly activated the change prediction.
        """
        if self.siamese_model is None:
            return np.zeros((t1_bands.shape[1], t1_bands.shape[2]), dtype=np.float32)

        t1_t = torch.from_numpy(t1_bands[:5]).unsqueeze(0).to(self.device)
        t2_t = torch.from_numpy(t2_bands[:5]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            f1_1, f1_2, f1_3, f1_4, f1_b = self.siamese_model.encoder(t1_t)
            f2_1, f2_2, f2_3, f2_4, f2_b = self.siamese_model.encoder(t2_t)

            # Deepest feature activation map
            diff_b = torch.abs(f1_b - f2_b)
            # Channel-wise average energy
            cam_low = torch.mean(diff_b, dim=1, keepdim=True)
            # Upsample to native raster resolution
            cam_high = F.interpolate(cam_low, size=(t1_bands.shape[1], t1_bands.shape[2]), mode="bilinear", align_corners=False)
            cam = cam_high.squeeze().cpu().numpy()

        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam
