"""
segmentation_dataset.py

Module 9: Forest Segmentation Using U-Net
Provides PyTorch Dataset and DataLoaders for full-scene dense semantic segmentation.
Prepares (5, H, W) satellite imagery and (1, H, W) binary forest masks.
"""

import os
import sys
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ForestSegmentationDataset(Dataset):
    """
    Loads paired multispectral satellite scenes and ground-truth forest masks.
    Extracts uniform tiles (e.g., 128x128) to ensure fast batch processing.
    """

    def __init__(self, image_paths: List[str], tile_size: int = 128, max_samples: Optional[int] = None):
        self.image_paths = image_paths
        self.tile_size = tile_size
        self.samples = []
        self._prepare_tiles()
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def _prepare_tiles(self):
        """Extracts 128x128 chips and ground truth forest masks."""
        for p in self.image_paths:
            with rasterio.open(p) as src:
                bands = src.read()  # (5, H, W)

            # Derive forest mask: NDVI >= 0.50 is Forest (1), else Non-Forest (0)
            red = bands[2]
            nir = bands[3]
            ndvi = (nir - red) / (nir + red + 1e-7)
            mask = (ndvi >= 0.50).astype(np.float32)

            _, h, w = bands.shape
            ts = self.tile_size

            # Slice into non-overlapping tiles
            for y in range(0, h - ts + 1, ts):
                for x in range(0, w - ts + 1, ts):
                    b_tile = bands[:, y:y + ts, x:x + ts].copy()
                    m_tile = mask[y:y + ts, x:x + ts][np.newaxis, :, :].copy()
                    self.samples.append((b_tile, m_tile))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        b_tile, m_tile = self.samples[idx]
        x = torch.tensor(b_tile, dtype=torch.float32)
        y = torch.tensor(m_tile, dtype=torch.float32)
        return x, y


def create_segmentation_dataloaders(
    dataset_root: str = "dataset",
    tile_size: int = 128,
    batch_size: int = 8,
    max_train_tiles: int = 24,
    max_val_tiles: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, validation, and test PyTorch DataLoaders for U-Net."""
    train_files = sorted(glob.glob(os.path.join(dataset_root, "train", "before", "*.tif")))
    val_files   = sorted(glob.glob(os.path.join(dataset_root, "validation", "before", "*.tif")))
    test_files  = sorted(glob.glob(os.path.join(dataset_root, "test", "before", "*.tif")))

    train_ds = ForestSegmentationDataset(train_files, tile_size=tile_size, max_samples=max_train_tiles)
    val_ds   = ForestSegmentationDataset(val_files, tile_size=tile_size, max_samples=max_val_tiles)
    test_ds  = ForestSegmentationDataset(test_files, tile_size=tile_size, max_samples=max_val_tiles)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
