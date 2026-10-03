"""
Module 8: CNN-Based Land Cover Classification
Deep Learning Foundation:
  - CNN fundamentals: Convolution, Feature Maps, Pooling
  - Custom Satellite Patch CNN with feature map extraction
  - ResNet Transfer Learning Backbone
  - 5-Class Land Cover Classification:
      0: Forest
      1: Bare Land / Deforested
      2: Agricultural Land
      3: Urban Area / Roads
      4: Water
"""

from .cnn_models import SatelliteLandCoverCNN, SatelliteResNet18
from .patch_dataset import LandCoverPatchDataset, create_dataloaders
from .trainer import train_land_cover_model, evaluate_model

__all__ = [
    "SatelliteLandCoverCNN",
    "SatelliteResNet18",
    "LandCoverPatchDataset",
    "create_dataloaders",
    "train_land_cover_model",
    "evaluate_model"
]
