"""
siamese_dataset.py

Module 11: Multi-Temporal Change Detection Using Siamese Neural Networks
Provides PyTorch Dataset and DataLoader generators for bi-temporal satellite image pairs:
  - T1: Pre-disturbance imagery (Before)
  - T2: Post-disturbance imagery (After)
  - Ground-truth change mask (1 = Deforestation, 0 = Unchanged)
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


class SiameseChangeDataset(Dataset):
    """
    Dataset for paired bi-temporal satellite scenes.
    Extracts aligned (5, H, W) chips for T1 and T2 along with (1, H, W) change masks.
    """

    def __init__(
        self,
        before_paths: List[str],
        tile_size: int = 128,
        max_samples: Optional[int] = None
    ):
        self.tile_size = tile_size
        self.samples = []
        self._load_and_tile_scenes(before_paths)

        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def _load_and_tile_scenes(self, before_paths: List[str]):
        """Matches before, after, and mask files and extracts tiles."""
        for b_path in before_paths:
            # Infer corresponding after and mask paths
            dir_name = os.path.dirname(b_path)
            base_filename = os.path.basename(b_path)
            split_dir = os.path.dirname(dir_name)  # e.g., dataset/train

            a_path = os.path.join(split_dir, "after", base_filename)
            m_path = os.path.join(split_dir, "masks", base_filename)

            if not os.path.exists(a_path) or not os.path.exists(m_path):
                continue

            with rasterio.open(b_path) as src_b:
                t1_img = src_b.read()  # (5, H, W)
            with rasterio.open(a_path) as src_a:
                t2_img = src_a.read()  # (5, H, W)
            with rasterio.open(m_path) as src_m:
                mask = src_m.read()    # (1, H, W)

            # Ensure proper channel dimension
            if mask.ndim == 2:
                mask = mask[np.newaxis, :, :]

            _, h, w = t1_img.shape
            ts = self.tile_size

            # Slice into uniform chips
            for y in range(0, h - ts + 1, ts):
                for x in range(0, w - ts + 1, ts):
                    t1_chip = t1_img[:, y:y + ts, x:x + ts].astype(np.float32)
                    t2_chip = t2_img[:, y:y + ts, x:x + ts].astype(np.float32)
                    mask_chip = mask[:, y:y + ts, x:x + ts].astype(np.float32)

                    self.samples.append((t1_chip, t2_chip, mask_chip))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t1_chip, t2_chip, mask_chip = self.samples[idx]
        return (
            torch.tensor(t1_chip, dtype=torch.float32),
            torch.tensor(t2_chip, dtype=torch.float32),
            torch.tensor(mask_chip, dtype=torch.float32)
        )


def create_siamese_dataloaders(
    dataset_root: str = "dataset",
    tile_size: int = 128,
    batch_size: int = 8,
    max_train_tiles: int = 32,
    max_val_tiles: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, validation, and test DataLoaders for Siamese Change Detection."""
    train_files = sorted(glob.glob(os.path.join(dataset_root, "train", "before", "*.tif")))
    val_files   = sorted(glob.glob(os.path.join(dataset_root, "validation", "before", "*.tif")))
    test_files  = sorted(glob.glob(os.path.join(dataset_root, "test", "before", "*.tif")))

    train_ds = SiameseChangeDataset(train_files, tile_size=tile_size, max_samples=max_train_tiles)
    val_ds   = SiameseChangeDataset(val_files, tile_size=tile_size, max_samples=max_val_tiles)
    test_ds  = SiameseChangeDataset(test_files, tile_size=tile_size, max_samples=max_val_tiles)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
