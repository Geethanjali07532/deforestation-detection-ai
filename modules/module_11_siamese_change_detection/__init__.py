"""
Module 11: Multi-Temporal Change Detection Using Siamese Neural Networks

Components:
  - SiameseChangeDataset: Multi-temporal paired dataset (T1 before, T2 after, deforestation mask)
  - SiameseUNetChangeDetector: Weight-sharing dual-stream encoder with multi-scale feature differencing
  - ContrastiveChangeLoss / CompoundChangeLoss: Handles extreme class imbalance in deforestation masks
  - SiameseTrainer & ChangeEvaluator: Training, validation, and metric computation
"""

from .siamese_dataset import SiameseChangeDataset, create_siamese_dataloaders
from .siamese_model import SiameseUNetChangeDetector
from .losses import CompoundChangeLoss
from .trainer import train_siamese_detector, evaluate_siamese_detector

__all__ = [
    "SiameseChangeDataset",
    "create_siamese_dataloaders",
    "SiameseUNetChangeDetector",
    "CompoundChangeLoss",
    "train_siamese_detector",
    "evaluate_siamese_detector"
]
