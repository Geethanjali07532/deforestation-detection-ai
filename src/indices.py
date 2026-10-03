"""
src/indices.py
Scientific Spectral Index Computations and Baseline Change Detection.
Computes NDVI, EVI, SAVI, NDWI, NBR, dNBR, RdNBR, and feature extraction for ML baselines.
"""

import numpy as np
from typing import Dict, Any, Tuple


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index (Rouse et al., 1974)."""
    denom = nir + red + 1e-7
    return np.clip((nir - red) / denom, -1.0, 1.0)


def compute_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray, g: float = 2.5, c1: float = 6.0, c2: float = 7.5, l: float = 1.0) -> np.ndarray:
    """Enhanced Vegetation Index (Huete et al., 2002) correcting for atmospheric haze and soil canopy background."""
    denom = nir + c1 * red - c2 * blue + l + 1e-7
    return np.clip(g * ((nir - red) / denom), -1.5, 1.5)


def compute_savi(nir: np.ndarray, red: np.ndarray, l: float = 0.5) -> np.ndarray:
    """Soil-Adjusted Vegetation Index (Huete, 1988) with soil brightness correction factor L=0.5."""
    denom = nir + red + l + 1e-7
    return np.clip(((nir - red) / denom) * (1.0 + l), -1.0, 1.0)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water / Moisture Index (Gao, 1996 / McFeeters, 1996)."""
    denom = green + nir + 1e-7
    return np.clip((green - nir) / denom, -1.0, 1.0)


def compute_nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio (Key & Benson, 2006) for wildfire and burn severity."""
    denom = nir + swir + 1e-7
    return np.clip((nir - swir) / denom, -1.0, 1.0)


def compute_all_spectral_indices(t1_bands: np.ndarray, t2_bands: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes all standard remote sensing vegetation, moisture, and burn indices
    from bi-temporal (5, H, W) rasters:
    Bands: 0=Blue, 1=Green, 2=Red, 3=NIR, 4=SWIR.
    """
    b1, g1, r1, n1, s1 = t1_bands[0], t1_bands[1], t1_bands[2], t1_bands[3], t1_bands[4]
    b2, g2, r2, n2, s2 = t2_bands[0], t2_bands[1], t2_bands[2], t2_bands[3], t2_bands[4]

    # Pre-disturbance (T1)
    ndvi1 = compute_ndvi(n1, r1)
    evi1 = compute_evi(n1, r1, b1)
    savi1 = compute_savi(n1, r1)
    ndwi1 = compute_ndwi(g1, n1)
    nbr1 = compute_nbr(n1, s1)

    # Post-disturbance (T2)
    ndvi2 = compute_ndvi(n2, r2)
    evi2 = compute_evi(n2, r2, b2)
    savi2 = compute_savi(n2, r2)
    ndwi2 = compute_ndwi(g2, n2)
    nbr2 = compute_nbr(n2, s2)

    # Differential Metrics
    dndvi = ndvi1 - ndvi2
    devi = evi1 - evi2
    dsavi = savi1 - savi2
    dndwi = ndwi1 - ndwi2
    dnbr = nbr1 - nbr2

    # Relativized Burn Ratios (Miller & Thode 2007)
    rdnbr = dnbr / np.sqrt(np.abs(nbr1) + 0.001)

    return {
        "ndvi_pre": ndvi1,
        "ndvi_post": ndvi2,
        "dndvi": dndvi,
        "evi_pre": evi1,
        "evi_post": evi2,
        "devi": devi,
        "savi_pre": savi1,
        "savi_post": savi2,
        "dsavi": dsavi,
        "ndwi_pre": ndwi1,
        "ndwi_post": ndwi2,
        "dndwi": dndwi,
        "nbr_pre": nbr1,
        "nbr_post": nbr2,
        "dnbr": dnbr,
        "rdnbr": rdnbr
    }


# Standard alias for backward and forward compatibility
compute_spectral_indices = compute_all_spectral_indices


def compute_baseline_change_mask(dndvi: np.ndarray, threshold: float = 0.30) -> np.ndarray:
    """
    Generates threshold-based baseline change mask from dNDVI.
    1 = Deforestation / Canopy Loss, 0 = Stable Forest / Background.
    """
    return (dndvi >= threshold).astype(np.uint8)


def extract_pixel_features_for_ml(t1_bands: np.ndarray, t2_bands: np.ndarray) -> np.ndarray:
    """
    Extracts 18 engineered spectral & differential features per pixel for Random Forest / XGBoost classifiers.
    Input shapes: (5, H, W)
    Output shape: (H*W, 18)
    """
    c, h, w = t1_bands.shape
    idx = compute_all_spectral_indices(t1_bands, t2_bands)

    b1, g1, r1, n1, s1 = [t1_bands[i].flatten() for i in range(5)]
    b2, g2, r2, n2, s2 = [t2_bands[i].flatten() for i in range(5)]

    features = np.column_stack([
        b1, g1, r1, n1, s1,
        b2, g2, r2, n2, s2,
        idx["ndvi_pre"].flatten(),
        idx["ndvi_post"].flatten(),
        idx["dndvi"].flatten(),
        idx["nbr_pre"].flatten(),
        idx["nbr_post"].flatten(),
        idx["dnbr"].flatten(),
        idx["evi_pre"].flatten(),
        idx["devi"].flatten()
    ])
    return features
