"""
patch_dataset.py

Module 8: CNN-Based Land Cover Classification
Creates PyTorch Dataset and DataLoaders for 5-class satellite image patches:
  Class 0: Forest
  Class 1: Bare Land / Deforested
  Class 2: Agricultural Land
  Class 3: Urban Area / Roads
  Class 4: Water
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class LandCoverPatchDataset(Dataset):
    """PyTorch Dataset holding multi-band patches and corresponding land cover labels."""

    CLASS_NAMES = [
        "Forest",
        "Bare Land / Deforested",
        "Agricultural Land",
        "Urban Area / Roads",
        "Water"
    ]

    def __init__(self, patches: np.ndarray, labels: np.ndarray, transform=None):
        """
        patches: float32 array of shape (N, C, H, W)
        labels: int64 array of shape (N,)
        """
        self.patches = torch.tensor(patches, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.patches[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def generate_synthetic_patch(
    class_id: int,
    patch_size: int = 32,
    num_bands: int = 5,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Synthesizes a realistic 5-band satellite patch (C, H, W) adhering to biophysical
    spectral profiles and spatial textures:
      - Bands: [0:Blue, 1:Green, 2:Red, 3:NIR, 4:SWIR]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Base spectral means [Blue, Green, Red, NIR, SWIR]
    profiles = {
        0: np.array([0.025, 0.065, 0.035, 0.520, 0.140]),  # Forest: High NIR, Low Red
        1: np.array([0.130, 0.190, 0.240, 0.310, 0.430]),  # Bare Soil / Deforested: High Red & SWIR
        2: np.array([0.040, 0.110, 0.070, 0.420, 0.190]),  # Agriculture: Intermediate vegetation
        3: np.array([0.160, 0.200, 0.220, 0.240, 0.330]),  # Urban / Roads: High visible & SWIR
        4: np.array([0.060, 0.050, 0.020, 0.010, 0.005]),  # Water: Near-zero NIR/SWIR
    }

    base = profiles[class_id]
    patch = np.zeros((num_bands, patch_size, patch_size), dtype=np.float32)

    # Class-specific spatial textures
    noise = rng.normal(0, 0.012, size=(num_bands, patch_size, patch_size)).astype(np.float32)

    if class_id == 2:  # Agriculture: periodic crop furrow rows
        y_coords = np.arange(patch_size)
        crop_rows = 0.04 * np.sin(y_coords / 2.0)[:, np.newaxis]
        noise[3] += crop_rows  # NIR row variation
    elif class_id == 3:  # Urban / Roads: linear high-contrast line
        road_y = patch_size // 2
        noise[:, road_y - 2:road_y + 3, :] += 0.06

    for b in range(num_bands):
        patch[b] = np.clip(base[b] + noise[b], 0.001, 0.999)

    return patch


def build_patch_dataset(
    total_samples: int = 1500,
    patch_size: int = 32,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a balanced dataset of satellite patches across all 5 classes."""
    rng = np.random.default_rng(seed)
    num_classes = 5
    per_class = total_samples // num_classes

    all_patches = []
    all_labels = []

    for c in range(num_classes):
        for _ in range(per_class):
            p = generate_synthetic_patch(c, patch_size=patch_size, rng=rng)
            all_patches.append(p)
            all_labels.append(c)

    X = np.stack(all_patches, axis=0)
    y = np.array(all_labels, dtype=np.int64)

    # Shuffle
    indices = np.arange(len(y))
    rng.shuffle(indices)
    return X[indices], y[indices]


def create_dataloaders(
    total_samples: int = 1500,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    patch_size: int = 32,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, validation, and test PyTorch DataLoaders."""
    X, y = build_patch_dataset(total_samples, patch_size=patch_size, seed=seed)

    n_total = len(y)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    train_ds = LandCoverPatchDataset(X_train, y_train)
    val_ds = LandCoverPatchDataset(X_val, y_val)
    test_ds = LandCoverPatchDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
