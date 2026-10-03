"""
dataset_generator.py

Generates realistic multispectral satellite imagery (GeoTIFF) with 5 key spectral bands:
  1. Blue (~490 nm)
  2. Green (~560 nm)
  3. Red (~665 nm)
  4. NIR - Near Infrared (~842 nm)
  5. SWIR - Short-Wave Infrared (~1610 nm)

The generated scene simulates a 10m-resolution Sentinel-2 observation over a tropical
forest region experiencing logging roads, expanding clearings (deforestation front),
bare soil, and a river basin.

Each land cover type adheres strictly to real-world physics and remote sensing reflectance
profiles (Surface Reflectance values from 0.0 to 1.0).
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import ColorInterp

def _generate_smooth_noise(height, width, scale=32, seed=42):
    """Generates smoothly interpolated 2D noise for natural vegetation heterogeneity."""
    rng = np.random.default_rng(seed)
    # Small grid of random values
    sh, sw = max(2, height // scale), max(2, width // scale)
    small_noise = rng.uniform(-1.0, 1.0, size=(sh, sw))
    
    # Bilinear upsample to full resolution
    y_coords = np.linspace(0, sh - 1, height)
    x_coords = np.linspace(0, sw - 1, width)
    
    y_low = np.floor(y_coords).astype(int)
    y_high = np.clip(y_low + 1, 0, sh - 1)
    x_low = np.floor(x_coords).astype(int)
    x_high = np.clip(x_low + 1, 0, sw - 1)
    
    y_frac = (y_coords - y_low)[:, np.newaxis]
    x_frac = (x_coords - x_low)[np.newaxis, :]
    
    top = (1 - x_frac) * small_noise[y_low, :][:, x_low] + x_frac * small_noise[y_low, :][:, x_high]
    bottom = (1 - x_frac) * small_noise[y_high, :][:, x_low] + x_frac * small_noise[y_high, :][:, x_high]
    
    noise = (1 - y_frac) * top + y_frac * bottom
    return noise

def create_sample_multispectral_geotiff(
    output_path: str = "data/samples/sentinel2_sample_scene.tif",
    width: int = 512,
    height: int = 512,
    resolution: float = 10.0,
    origin_x: float = 500000.0,
    origin_y: float = 9700000.0,
    epsg_code: int = 32620,
    seed: int = 42
) -> str:
    """
    Creates and saves a georeferenced 5-band GeoTIFF with realistic spectral signatures.

    Bands:
      Band 1: Blue  (490 nm)
      Band 2: Green (560 nm)
      Band 3: Red   (665 nm)
      Band 4: NIR   (842 nm)
      Band 5: SWIR  (1610 nm)

    Simulated Features:
      - Dense Canopy Rainforest
      - Logging / Access Roads
      - Deforested / Cleared Patches (Slash & Bare Dry Soil)
      - Meandering River / Water Body
      - Degraded Forest Edge (Transition Zone)

    Returns:
      Absolute path to the created GeoTIFF.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. Base canopy noise layers
    canopy_var = _generate_smooth_noise(height, width, scale=24, seed=seed) * 0.03
    fine_noise = rng.normal(0, 0.008, size=(height, width))

    # Initialize land cover segmentation mask:
    # 0 = Dense Forest, 1 = Cleared / Bare Soil, 2 = Water, 3 = Road, 4 = Degraded Forest
    mask = np.zeros((height, width), dtype=np.uint8)

    # 2. Add Meandering River (Water)
    x = np.arange(width)
    y_center = (height * 0.75 + 40 * np.sin(x / 40.0) + 15 * np.cos(x / 20.0)).astype(int)
    for i, yc in enumerate(y_center):
        y_min = max(0, yc - 9)
        y_max = min(height, yc + 10)
        mask[y_min:y_max, i] = 2

    # 3. Add Deforestation Clearing Patches (Polygonal & irregular clearings)
    # Clearing cluster 1 (North-West)
    yy, xx = np.ogrid[:height, :width]
    clearing_1 = ((xx - 140)**2 / (65**2) + (yy - 130)**2 / (45**2) <= 1)
    mask[clearing_1] = 1

    # Clearing cluster 2 (Center-East active clearing front)
    clearing_2 = (
        (xx >= 280) & (xx <= 420) &
        (yy >= 160) & (yy <= 310) &
        ((xx - 280) + (yy - 160) < 220)
    )
    mask[clearing_2] = 1

    # Clearing cluster 3 (Small agricultural expansion block)
    clearing_3 = ((xx - 380)**2 / (50**2) + (yy - 80)**2 / (35**2) <= 1)
    mask[clearing_3] = 1

    # 4. Add Logging Road (linear feature penetrating forest into clearings)
    road_x = np.arange(width)
    road_y = (180 + 0.15 * road_x + 8 * np.sin(road_x / 30.0)).astype(int)
    for i, ry in enumerate(road_y):
        if 0 <= ry < height:
            y_min = max(0, ry - 2)
            y_max = min(height, ry + 3)
            mask[y_min:y_max, i] = 3

    # Branch road to clearing 1
    branch_y = np.arange(130, 200)
    branch_x = (140 + 0.3 * (branch_y - 130) + 2 * np.sin(branch_y / 10.0)).astype(int)
    for j, bx in enumerate(branch_x):
        if 0 <= bx < width and 0 <= branch_y[j] < height:
            mask[branch_y[j], max(0, bx-2):min(width, bx+3)] = 3

    # 5. Degraded Forest / Forest Edge Buffer (surrounding clearings)
    from scipy.ndimage import binary_dilation
    cleared_binary = (mask == 1)
    degraded_buffer = binary_dilation(cleared_binary, iterations=12) & (~cleared_binary) & (mask != 2) & (mask != 3)
    mask[degraded_buffer] = 4

    # 6. Synthesize Spectral Reflectance (0.0 to 1.0) based on realistic physics:
    # Signature Profiles: [Blue, Green, Red, NIR, SWIR]
    # Dense Forest: Chlorophyll absorbs Blue & Red, reflects slight Green, massive cellular NIR scattering, high water absorption in SWIR
    forest_mean = np.array([0.025, 0.065, 0.035, 0.520, 0.140])
    
    # Degraded Forest: Reduced leaf area index, lower NIR, higher Red & SWIR
    degraded_mean = np.array([0.045, 0.085, 0.080, 0.360, 0.230])

    # Cleared / Bare Soil: High Red, dry surface with high SWIR, moderate NIR
    soil_mean = np.array([0.130, 0.190, 0.240, 0.310, 0.430])

    # Road (compacted gravel/dirt): Moderate-high visible & SWIR, low NIR contrast
    road_mean = np.array([0.160, 0.200, 0.220, 0.240, 0.330])

    # Water (River): Absorbs NIR and SWIR almost completely, modest visible reflectance
    water_mean = np.array([0.060, 0.050, 0.020, 0.010, 0.005])

    # Allocate 5-band raster (bands, height, width)
    bands = np.zeros((5, height, width), dtype=np.float32)

    profiles = {
        0: forest_mean,
        1: soil_mean,
        2: water_mean,
        3: road_mean,
        4: degraded_mean
    }

    for class_id, profile in profiles.items():
        class_pixels = (mask == class_id)
        for b_idx in range(5):
            base_val = profile[b_idx]
            # Add spatial texture variation
            tex = canopy_var * (0.8 if class_id in (0, 4) else 0.3)
            val = base_val + tex + fine_noise
            bands[b_idx, class_pixels] = val[class_pixels]

    # Clip reflectance to valid physical limits [0.001, 0.999]
    bands = np.clip(bands, 0.001, 0.999).astype(np.float32)

    # 7. Write GeoTIFF with complete georeferencing and band metadata
    transform = from_origin(origin_x, origin_y, resolution, resolution)
    crs = f"EPSG:{epsg_code}"

    band_names = [
        "Band 1 - Blue (490 nm)",
        "Band 2 - Green (560 nm)",
        "Band 3 - Red (665 nm)",
        "Band 4 - NIR (842 nm)",
        "Band 5 - SWIR (1610 nm)"
    ]

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=5,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=-9999.0
    ) as dst:
        for b_idx in range(5):
            dst.write(bands[b_idx], b_idx + 1)
            dst.set_band_description(b_idx + 1, band_names[b_idx])

    return os.path.abspath(output_path)

if __name__ == "__main__":
    path = create_sample_multispectral_geotiff()
    print(f"Sample GeoTIFF created successfully at: {path}")
