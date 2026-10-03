"""
dataset_builder.py

Module 3: Dataset Collection & Organization
Builds a complete, authentic multi-temporal satellite dataset for deforestation detection:
  - Generates realistic Before and After satellite observations (GeoTIFFs with Red, Green, Blue, NIR, SWIR bands).
  - Simulates genuine temporal changes: logging roads, slash clearing fronts, and agricultural expansion.
  - Generates exact ground-truth binary masks (1 = Deforested, 0 = Stable / No Change).
  - Populates standard train / validation / test splits.
  - Compiles a dataset manifest (metadata.csv) tracking change metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def _smooth_noise(height, width, scale=24, rng=None):
    """Generates 2D interpolated fractal noise for canopy heterogeneity."""
    if rng is None:
        rng = np.random.default_rng(42)
    sh, sw = max(2, height // scale), max(2, width // scale)
    small = rng.uniform(-1.0, 1.0, size=(sh, sw))

    y_coords = np.linspace(0, sh - 1, height)
    x_coords = np.linspace(0, sw - 1, width)
    y_low = np.floor(y_coords).astype(int)
    y_high = np.clip(y_low + 1, 0, sh - 1)
    x_low = np.floor(x_coords).astype(int)
    x_high = np.clip(x_low + 1, 0, sw - 1)

    y_frac = (y_coords - y_low)[:, np.newaxis]
    x_frac = (x_coords - x_low)[np.newaxis, :]

    top = (1 - x_frac) * small[y_low, :][:, x_low] + x_frac * small[y_low, :][:, x_high]
    bottom = (1 - x_frac) * small[y_high, :][:, x_low] + x_frac * small[y_high, :][:, x_high]
    return (1 - y_frac) * top + y_frac * bottom

def generate_scene_pair(
    height: int = 256,
    width: int = 256,
    seed: int = 42,
    deforestation_rate: float = 0.15
):
    """
    Synthesizes a realistic temporal pair (Before Image, After Image, Ground Truth Mask).
    
    Before: Intact tropical forest, meandering river, baseline soil/clearing.
    After: Same geographic scene with newly introduced deforestation fronts and logging tracks.
    Mask: 1 = Newly Deforested Pixels, 0 = Unchanged / Background.
    """
    rng = np.random.default_rng(seed)

    # 1. Base Landscape & River
    yy, xx = np.ogrid[:height, :width]
    river_center = (height * 0.7 + 25 * np.sin(xx / 30.0) + 10 * np.cos(xx / 15.0)).astype(int)[0]
    river_mask = np.zeros((height, width), dtype=bool)
    for i, rc in enumerate(river_center):
        y_min = max(0, rc - 5)
        y_max = min(height, rc + 6)
        river_mask[y_min:y_max, i] = True

    # 2. Existing baseline clearings in 'Before'
    baseline_clearings = np.zeros((height, width), dtype=bool)
    num_baseline = rng.integers(1, 3)
    for _ in range(num_baseline):
        cx, cy = rng.integers(30, width - 30), rng.integers(30, height - 30)
        rx, ry = rng.integers(15, 30), rng.integers(12, 25)
        patch = ((xx - cx)**2 / (rx**2) + (yy - cy)**2 / (ry**2) <= 1) & (~river_mask)
        baseline_clearings |= patch

    # 3. New Deforestation events occurring in 'After'
    # Combination of expanding clearings and logging roads
    new_deforestation = np.zeros((height, width), dtype=bool)
    num_new = rng.integers(2, 5)
    for _ in range(num_new):
        cx = rng.integers(40, width - 40)
        cy = rng.integers(40, height - 40)
        rx = rng.integers(20, int(35 + 20 * deforestation_rate))
        ry = rng.integers(15, int(30 + 15 * deforestation_rate))
        # Add angular/polygonal clearing shape typical of cattle ranching/agriculture
        shape_noise = rng.uniform(0.7, 1.3, size=(height, width))
        patch = (((xx - cx)**2 / (rx**2) + (yy - cy)**2 / (ry**2)) * shape_noise <= 1) & (~river_mask) & (~baseline_clearings)
        new_deforestation |= patch

    # Add logging road penetrating the forest to the clearing
    road_start_y = rng.integers(height // 4, 3 * height // 4)
    road_x = np.arange(width)
    road_y = (road_start_y + 0.1 * road_x + 6 * np.sin(road_x / 20.0)).astype(int)
    road_mask = np.zeros((height, width), dtype=bool)
    for i, ry in enumerate(road_y):
        if 0 <= ry < height:
            road_mask[max(0, ry - 1):min(height, ry + 2), i] = True
    road_mask &= (~river_mask) & (~baseline_clearings)
    new_deforestation |= road_mask

    # Ground truth change mask (1 = newly deforested, 0 = unchanged)
    mask = new_deforestation.astype(np.uint8)

    # 4. Synthesize 5 spectral bands: [Blue, Green, Red, NIR, SWIR]
    # Realistic reflectance profiles:
    forest_prof = np.array([0.025, 0.065, 0.035, 0.520, 0.140])
    soil_prof   = np.array([0.130, 0.190, 0.240, 0.310, 0.430])
    water_prof  = np.array([0.060, 0.050, 0.020, 0.010, 0.005])

    canopy_var = _smooth_noise(height, width, scale=20, rng=rng) * 0.025
    noise_before = rng.normal(0, 0.008, size=(height, width))
    noise_after = rng.normal(0, 0.008, size=(height, width))

    before_bands = np.zeros((5, height, width), dtype=np.float32)
    after_bands = np.zeros((5, height, width), dtype=np.float32)

    for b in range(5):
        # Baseline 'Before' state:
        b_val = np.full((height, width), forest_prof[b], dtype=np.float32)
        b_val[baseline_clearings] = soil_prof[b]
        b_val[river_mask] = water_prof[b]
        before_bands[b] = np.clip(b_val + canopy_var + noise_before, 0.001, 0.999)

        # 'After' state:
        a_val = b_val.copy()
        a_val[new_deforestation] = soil_prof[b]  # Forest converted to bare soil / slash
        after_bands[b] = np.clip(a_val + canopy_var + noise_after, 0.001, 0.999)

    return before_bands, after_bands, mask

def save_geotiff(filepath: str, data: np.ndarray, transform, crs="EPSG:32620", nodata=-9999.0):
    """Writes a multi-band float32 GeoTIFF or single-band mask GeoTIFF."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if data.ndim == 2:
        count = 1
        height, width = data.shape
        data = data[np.newaxis, :, :]
        dtype = data.dtype
    else:
        count, height, width = data.shape
        dtype = data.dtype

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        for i in range(count):
            dst.write(data[i], i + 1)

def build_multitemporal_dataset(
    dataset_root: str = "dataset",
    train_count: int = 16,
    val_count: int = 4,
    test_count: int = 4,
    tile_size: int = 256,
    pixel_res: float = 10.0
):
    """
    Builds the complete multi-temporal dataset hierarchy:
      dataset/
      ├── train/ (before/, after/, masks/)
      ├── validation/ (before/, after/, masks/)
      └── test/ (before/, after/, masks/)
    Generates full paired samples and outputs a master metadata.csv.
    """
    splits_spec = [
        ("train", train_count, 100),
        ("validation", val_count, 200),
        ("test", test_count, 300)
    ]

    records = []
    origin_x, origin_y = 500000.0, 9700000.0
    transform = from_origin(origin_x, origin_y, pixel_res, pixel_res)

    print("=" * 70)
    print("🛰️  BUILDING MULTI-TEMPORAL DEFORESTATION DATASET (MODULE 3)")
    print("=" * 70)

    for split, count, seed_offset in splits_spec:
        print(f"\n📦 Generating split [{split.upper()}]: {count} paired observations...")
        for idx in range(1, count + 1):
            sample_id = f"scene_{split}_{idx:03d}"
            seed = seed_offset + idx

            before_bands, after_bands, mask = generate_scene_pair(
                height=tile_size,
                width=tile_size,
                seed=seed,
                deforestation_rate=0.10 + 0.05 * (idx % 3)
            )

            before_path = os.path.join(dataset_root, split, "before", f"{sample_id}.tif")
            after_path = os.path.join(dataset_root, split, "after", f"{sample_id}.tif")
            mask_path = os.path.join(dataset_root, split, "masks", f"{sample_id}.tif")

            save_geotiff(before_path, before_bands, transform)
            save_geotiff(after_path, after_bands, transform)
            save_geotiff(mask_path, mask.astype(np.uint8), transform, nodata=255)

            deforest_pixels = int(np.sum(mask == 1))
            total_pixels = tile_size * tile_size
            deforest_pct = (deforest_pixels / total_pixels) * 100.0
            # 10m x 10m pixel = 100 m^2 = 0.01 hectares
            deforest_area_ha = deforest_pixels * 0.01

            records.append({
                "sample_id": sample_id,
                "split": split,
                "before_path": os.path.abspath(before_path),
                "after_path": os.path.abspath(after_path),
                "mask_path": os.path.abspath(mask_path),
                "width": tile_size,
                "height": tile_size,
                "bands": 5,
                "resolution_m": pixel_res,
                "deforest_pixels": deforest_pixels,
                "deforest_pct": round(deforest_pct, 2),
                "deforest_area_ha": round(deforest_area_ha, 2)
            })

    # Save dataset manifest
    manifest_path = os.path.join(dataset_root, "metadata.csv")
    df = pd.DataFrame(records)
    df.to_csv(manifest_path, index=False)
    print("\n" + "=" * 70)
    print(f"✅ Successfully created {len(records)} paired multi-temporal sets!")
    print(f"📁 Dataset root: {os.path.abspath(dataset_root)}")
    print(f"📋 Master manifest: {os.path.abspath(manifest_path)}")
    print("=" * 70)
    return manifest_path

if __name__ == "__main__":
    build_multitemporal_dataset()
