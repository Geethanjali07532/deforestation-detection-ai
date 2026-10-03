"""
preprocessing_pipeline.py

Module 4: Satellite Image Preprocessing Pipeline
Implements the full preprocessing chain for multi-temporal satellite imagery:
  Raw Satellite Image
         ↓
  Cloud / Noise Removal
         ↓
  Geometric Alignment (Co-Registration)
         ↓
  Band Selection & Stacking
         ↓
  Normalization
         ↓
  Image Tiles (Chips)
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from scipy.ndimage import median_filter, shift
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class CloudMasker:
    """
    Detects cloud contamination and cloud shadows using multispectral band thresholds.
    In Sentinel-2 / Landsat, clouds exhibit high visible reflectance (Blue > 0.25)
    coupled with distinct NIR/SWIR ratios.
    """

    def __init__(self, blue_threshold: float = 0.35, high_refl_threshold: float = 0.40):
        self.blue_thresh = blue_threshold
        self.high_thresh = high_refl_threshold

    def create_cloud_mask(self, bands: np.ndarray) -> np.ndarray:
        """
        Input: bands array of shape (5, H, W) [0:Blue, 1:Green, 2:Red, 3:NIR, 4:SWIR]
        Returns: binary mask (H, W) where 1 = Cloud / Contaminated, 0 = Clear Sky
        """
        blue = bands[0]
        green = bands[1]
        red = bands[2]
        # Clouds are intensely bright across all visible channels
        visible_mean = (blue + green + red) / 3.0
        cloud_mask = (blue > self.blue_thresh) & (visible_mean > self.high_thresh)
        return cloud_mask.astype(np.uint8)

    def mask_and_interpolate(self, bands: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
        """Applies cloud mask and smooths contaminated regions."""
        cleaned = bands.copy()
        for b in range(bands.shape[0]):
            band = cleaned[b]
            if np.any(cloud_mask):
                # Replace cloud pixels with neighborhood median
                med = median_filter(band, size=7)
                band[cloud_mask == 1] = med[cloud_mask == 1]
            cleaned[b] = band
        return cleaned


class NoiseFilter:
    """
    Removes atmospheric speckle and sensor noise using edge-preserving filtering.
    Preserves crisp linear boundaries of logging roads and deforestation fronts.
    """

    def __init__(self, filter_size: int = 3):
        self.filter_size = filter_size

    def apply(self, bands: np.ndarray) -> np.ndarray:
        """Applies median filtering across each band independently."""
        filtered = np.zeros_like(bands)
        for b in range(bands.shape[0]):
            filtered[b] = median_filter(bands[b], size=self.filter_size)
        return filtered


class GeometricAligner:
    """
    Sub-pixel and pixel-level Co-Registration between Before and After images.
    Uses Phase Cross-Correlation on the Near-Infrared / Green band to detect
    geometric drift and align temporal observations perfectly.
    """

    def __init__(self, reference_band_idx: int = 3):  # Band 3 is NIR
        self.ref_idx = reference_band_idx

    def compute_shift(self, before_band: np.ndarray, after_band: np.ndarray) -> Tuple[float, float]:
        """Calculates (dy, dx) pixel translation using phase correlation."""
        # Use single-precision float for 2D FFT cross-correlation
        f0 = np.fft.fft2(before_band)
        f1 = np.fft.fft2(after_band)
        eps = 1e-12
        cross_power = (f0 * f1.conj()) / (np.abs(f0 * f1.conj()) + eps)
        r = np.fft.ifft2(cross_power)
        r = np.fft.fftshift(r)
        
        mid_y, mid_x = r.shape[0] // 2, r.shape[1] // 2
        y_max, x_max = np.unravel_index(np.argmax(np.abs(r)), r.shape)
        dy = y_max - mid_y
        dx = x_max - mid_x
        return float(dy), float(dx)

    def align(self, before_bands: np.ndarray, after_bands: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """
        Aligns the After image to match the spatial geometry of the Before image.
        Returns: (before_bands, aligned_after_bands, (dy, dx))
        """
        ref_b = before_bands[self.ref_idx]
        ref_a = after_bands[self.ref_idx]
        dy, dx = self.compute_shift(ref_b, ref_a)

        aligned_after = np.zeros_like(after_bands)
        for b in range(after_bands.shape[0]):
            aligned_after[b] = shift(after_bands[b], shift=(-dy, -dx), mode="nearest")

        return before_bands, aligned_after, (dy, dx)


class BandStacker:
    """
    Selects designated spectral bands and creates temporal band stacks
    for early-fusion change detection models.
    """

    def __init__(self, selected_bands: Optional[List[int]] = None):
        # Default: all 5 bands [0:Blue, 1:Green, 2:Red, 3:NIR, 4:SWIR]
        self.selected_bands = selected_bands or [0, 1, 2, 3, 4]

    def select(self, bands: np.ndarray) -> np.ndarray:
        """Extracts specified band indices."""
        return bands[self.selected_bands]

    def stack_temporal_pair(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """
        Concatenates Before and After along channel axis.
        Shape: (C_before + C_after, H, W).
        For 5 bands each, outputs a 10-channel tensor.
        """
        b_sel = self.select(before)
        a_sel = self.select(after)
        return np.concatenate([b_sel, a_sel], axis=0)


class RasterNormalizer:
    """
    Radiometric normalization:
      - 'robust': 2%-98% percentile linear stretch to [0, 1]
      - 'minmax': standard min-max scaling to [0, 1]
      - 'zscore': zero-mean unit-variance standardization
    """

    def __init__(self, method: str = "robust", p_min: float = 2.0, p_max: float = 98.0):
        self.method = method
        self.p_min = p_min
        self.p_max = p_max

    def normalize(self, data: np.ndarray) -> np.ndarray:
        normed = np.zeros_like(data, dtype=np.float32)
        for c in range(data.shape[0]):
            channel = data[c]
            if self.method == "robust":
                v_min, v_max = np.percentile(channel, (self.p_min, self.p_max))
                if v_max > v_min:
                    normed[c] = np.clip((channel - v_min) / (v_max - v_min), 0.0, 1.0)
                else:
                    normed[c] = np.clip(channel, 0.0, 1.0)
            elif self.method == "minmax":
                v_min, v_max = np.min(channel), np.max(channel)
                normed[c] = (channel - v_min) / (v_max - v_min + 1e-8)
            elif self.method == "zscore":
                mean, std = np.mean(channel), np.std(channel)
                normed[c] = (channel - mean) / (std + 1e-8)
            else:
                normed[c] = channel
        return normed


class RasterTiler:
    """
    Tiles large satellite imagery and masks into uniform ML-ready chips
    (e.g., 128x128 or 256x256) with configurable stride/overlap.
    """

    def __init__(self, tile_size: int = 128, stride: int = 128):
        self.tile_size = tile_size
        self.stride = stride

    def tile_pair_and_mask(
        self,
        before: np.ndarray,
        after: np.ndarray,
        mask: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Extracts synchronized chips from Before, After, and Mask rasters.
        Returns a list of dicts with chipped arrays and bounding coordinates.
        """
        _, h, w = before.shape
        chips = []
        chip_id = 0

        for y in range(0, h - self.tile_size + 1, self.stride):
            for x in range(0, w - self.tile_size + 1, self.stride):
                b_chip = before[:, y:y + self.tile_size, x:x + self.tile_size]
                a_chip = after[:, y:y + self.tile_size, x:x + self.tile_size]
                m_chip = mask[y:y + self.tile_size, x:x + self.tile_size]

                chips.append({
                    "chip_id": chip_id,
                    "row_start": y,
                    "col_start": x,
                    "before": b_chip,
                    "after": a_chip,
                    "mask": m_chip,
                    "deforested_pixels": int(np.sum(m_chip == 1)),
                    "deforested_pct": float((np.sum(m_chip == 1) / m_chip.size) * 100.0)
                })
                chip_id += 1

        return chips


class SatellitePreprocessingPipeline:
    """
    End-to-End Satellite Image Preprocessing Pipeline (Module 4).
    Ties together:
      Raw -> Cloud Masking -> Noise Filter -> Co-Registration -> Band Stacking -> Normalization -> Tiling
    """

    def __init__(
        self,
        tile_size: int = 128,
        stride: int = 128,
        normalization_method: str = "robust",
        selected_bands: Optional[List[int]] = None
    ):
        self.cloud_masker = CloudMasker()
        self.noise_filter = NoiseFilter(filter_size=3)
        self.aligner = GeometricAligner(reference_band_idx=3)
        self.band_stacker = BandStacker(selected_bands=selected_bands)
        self.normalizer = RasterNormalizer(method=normalization_method)
        self.tiler = RasterTiler(tile_size=tile_size, stride=stride)

    def process_pair(
        self,
        raw_before: np.ndarray,
        raw_after: np.ndarray,
        raw_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Executes full preprocessing stages on a multi-temporal pair.
        Returns intermediate products and final ready-to-train chips.
        """
        stages = {}

        # 1. Cloud Masking
        cloud_b = self.cloud_masker.create_cloud_mask(raw_before)
        cloud_a = self.cloud_masker.create_cloud_mask(raw_after)
        clean_b = self.cloud_masker.mask_and_interpolate(raw_before, cloud_b)
        clean_a = self.cloud_masker.mask_and_interpolate(raw_after, cloud_a)
        stages["cloud_mask_before"] = cloud_b
        stages["cloud_mask_after"] = cloud_a

        # 2. Noise Removal
        denoised_b = self.noise_filter.apply(clean_b)
        denoised_a = self.noise_filter.apply(clean_a)
        stages["denoised_before"] = denoised_b
        stages["denoised_after"] = denoised_a

        # 3. Geometric Alignment / Co-Registration
        aligned_b, aligned_a, shift_detected = self.aligner.align(denoised_b, denoised_a)
        stages["aligned_before"] = aligned_b
        stages["aligned_after"] = aligned_a
        stages["detected_shift_px"] = shift_detected

        # 4. Band Selection
        sel_b = self.band_stacker.select(aligned_b)
        sel_a = self.band_stacker.select(aligned_a)

        # 5. Normalization
        norm_b = self.normalizer.normalize(sel_b)
        norm_a = self.normalizer.normalize(sel_a)
        stages["normalized_before"] = norm_b
        stages["normalized_after"] = norm_a

        # 6. Temporal Band Stacking (e.g. 10-channel tensor)
        stacked_10ch = self.band_stacker.stack_temporal_pair(norm_b, norm_a)
        stages["stacked_temporal"] = stacked_10ch

        # 7. Image Tiling / Chip Extraction
        chips = self.tiler.tile_pair_and_mask(norm_b, norm_a, raw_mask)
        stages["chips"] = chips
        stages["total_chips_extracted"] = len(chips)

        return stages
