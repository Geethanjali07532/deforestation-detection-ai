"""
spectral_inspector.py

Comprehensive Satellite Imagery & Spectral Band Inspector for Module 2.
Accomplishes:
  1. GeoTIFF loading and raster metadata extraction via rasterio.
  2. Inspection of Red, Green, Blue, NIR, and SWIR spectral bands.
  3. Calculation of statistical profiles per band (min, max, mean, std, percentiles).
  4. Generation of True Color (RGB) and False Color (CIR, SWIR-Agriculture) composites.
  5. Spectral signature curve plotting (Forest vs. Cleared Land vs. Water vs. Roads).
  6. Scientific explanation of why vegetation behaves uniquely across the electromagnetic spectrum.
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import rasterio

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

class SatelliteImageryInspector:
    """Inspector and analysis engine for multispectral satellite rasters."""

    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Satellite raster not found at: {filepath}")
        self.filepath = filepath
        self.dataset = rasterio.open(filepath)
        self.meta = self.dataset.meta.copy()
        self.bounds = self.dataset.bounds
        self.crs = self.dataset.crs
        self.transform = self.dataset.transform
        self.band_count = self.dataset.count
        self.width = self.dataset.width
        self.height = self.dataset.height

        # Read all bands into numpy array (shape: bands, height, width)
        self.bands_data = self.dataset.read()

        # Identify bands (assuming 1=Blue, 2=Green, 3=Red, 4=NIR, 5=SWIR)
        # Note: rasterio band indexing is 1-based, numpy array is 0-based
        self.blue = self.bands_data[0]
        self.green = self.bands_data[1]
        self.red = self.bands_data[2]
        self.nir = self.bands_data[3] if self.band_count >= 4 else None
        self.swir = self.bands_data[4] if self.band_count >= 5 else None

        self.band_dict = {
            "Blue (490 nm)": self.blue,
            "Green (560 nm)": self.green,
            "Red (665 nm)": self.red,
        }
        if self.nir is not None:
            self.band_dict["NIR (842 nm)"] = self.nir
        if self.swir is not None:
            self.band_dict["SWIR (1610 nm)"] = self.swir

    def get_metadata_report(self) -> Dict[str, Any]:
        """Returns structured metadata regarding projection, resolution, and raster grid."""
        res_x, res_y = self.transform[0], abs(self.transform[4])
        report = {
            "File Path": self.filepath,
            "Driver": self.meta.get("driver"),
            "Dimensions": f"{self.width} cols x {self.height} rows",
            "Number of Bands": self.band_count,
            "Data Type": self.meta.get("dtype"),
            "CRS (Coordinate Reference System)": str(self.crs),
            "Spatial Resolution": f"{res_x:.2f} m x {res_y:.2f} m per pixel",
            "Bounding Box": {
                "West (Min X)": self.bounds.left,
                "South (Min Y)": self.bounds.bottom,
                "East (Max X)": self.bounds.right,
                "North (Max Y)": self.bounds.top,
            },
            "NoData Value": self.meta.get("nodata"),
            "Band Descriptions": [self.dataset.descriptions[i] or f"Band {i+1}" for i in range(self.band_count)]
        }
        return report

    def print_metadata(self):
        """Prints formatted metadata report to console."""
        print("=" * 65)
        print("📡 SATELLITE RASTER METADATA INSPECTION")
        print("=" * 65)
        report = self.get_metadata_report()
        for k, v in report.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for sub_k, sub_v in v.items():
                    print(f"  • {sub_k}: {sub_v}")
            elif isinstance(v, list):
                print(f"{k}:")
                for item in v:
                    print(f"  • {item}")
            else:
                print(f"{k}: {v}")
        print("=" * 65)

    def compute_band_statistics(self) -> Dict[str, Dict[str, float]]:
        """Computes statistical metrics (min, max, mean, std, percentiles) for each band."""
        stats = {}
        for name, data in self.band_dict.items():
            valid_pixels = data[data != self.meta.get("nodata", -9999.0)]
            stats[name] = {
                "min": float(np.min(valid_pixels)),
                "max": float(np.max(valid_pixels)),
                "mean": float(np.mean(valid_pixels)),
                "std": float(np.std(valid_pixels)),
                "p2": float(np.percentile(valid_pixels, 2)),
                "p50_median": float(np.median(valid_pixels)),
                "p98": float(np.percentile(valid_pixels, 98)),
            }
        return stats

    def print_band_statistics(self):
        """Prints statistical table to console."""
        print("\n" + "=" * 80)
        print(f"{'BAND NAME':<18} | {'MIN':<7} | {'MAX':<7} | {'MEAN':<7} | {'STD':<7} | {'MEDIAN':<7} | {'P2-P98 RANGE':<14}")
        print("-" * 80)
        stats = self.compute_band_statistics()
        for name, s in stats.items():
            p_range = f"[{s['p2']:.3f}, {s['p98']:.3f}]"
            print(f"{name:<18} | {s['min']:<7.3f} | {s['max']:<7.3f} | {s['mean']:<7.3f} | {s['std']:<7.3f} | {s['p50_median']:<7.3f} | {p_range:<14}")
        print("=" * 80)

    @staticmethod
    def _stretch(band: np.ndarray, p_min: float = 2.0, p_max: float = 98.0) -> np.ndarray:
        """Applies 2-98% percentile linear contrast stretch to normalize band to [0, 1]."""
        v_min, v_max = np.percentile(band, (p_min, p_max))
        if v_max <= v_min:
            return np.clip(band, 0.0, 1.0)
        stretched = (band - v_min) / (v_max - v_min)
        return np.clip(stretched, 0.0, 1.0)

    def create_rgb_composite(self) -> np.ndarray:
        """True Color Composite (R: Red, G: Green, B: Blue) - As the human eye sees."""
        r = self._stretch(self.red)
        g = self._stretch(self.green)
        b = self._stretch(self.blue)
        return np.stack([r, g, b], axis=-1)

    def create_false_color_cir(self) -> np.ndarray:
        """
        Color Infrared (CIR) False Color Composite (R: NIR, G: Red, B: Green).
        - Healthy forest canopy reflects massive NIR energy and appears bright red / crimson.
        - Cleared land, bare soil, and urban structures appear cyan / grey.
        - Water bodies absorb NIR completely and appear dark blue / black.
        """
        if self.nir is None:
            raise ValueError("NIR band required for False Color CIR composite.")
        r = self._stretch(self.nir)
        g = self._stretch(self.red)
        b = self._stretch(self.green)
        return np.stack([r, g, b], axis=-1)

    def create_swir_composite(self) -> np.ndarray:
        """
        SWIR / Agriculture Composite (R: SWIR, G: NIR, B: Red).
        - Dense moisture-rich vegetation appears vibrant green.
        - Deforested land, clearings, and dry bare ground appear bright orange / copper.
        - Water bodies appear deep blue / black.
        - Extremely sensitive to moisture stress and canopy loss!
        """
        if self.swir is None or self.nir is None:
            raise ValueError("SWIR and NIR bands required for SWIR composite.")
        r = self._stretch(self.swir)
        g = self._stretch(self.nir)
        b = self._stretch(self.red)
        return np.stack([r, g, b], axis=-1)

    def plot_individual_bands(self, output_path: Optional[str] = None):
        """Plots grayscale views of each individual spectral band side-by-side with colorbars."""
        num_bands = len(self.band_dict)
        fig, axes = plt.subplots(1, num_bands, figsize=(4.2 * num_bands, 4.5), constrained_layout=True)

        for ax, (b_name, b_data) in zip(axes, self.band_dict.items()):
            im = ax.imshow(b_data, cmap="gray", vmin=0.0, vmax=float(np.percentile(b_data, 99.5)))
            ax.set_title(b_name, fontsize=12, fontweight="bold", pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Reflectance", fontsize=9)

        fig.suptitle(
            "Individual Spectral Bands (Raw Surface Reflectance)\n"
            "Notice how forest is dark in Red/Blue (chlorophyll absorption) and luminous in NIR (cell scattering)",
            fontsize=13, fontweight="bold", y=1.05
        )

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f" Saved individual bands plot to: {output_path}")
        plt.close(fig)

    def plot_composites(self, output_path: Optional[str] = None):
        """Compares True Color RGB vs False Color CIR vs SWIR Composite."""
        rgb = self.create_rgb_composite()
        cir = self.create_false_color_cir()
        swir = self.create_swir_composite()

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

        # 1. True Color
        axes[0].imshow(rgb)
        axes[0].set_title("True Color Composite (RGB)\nRed: Band 3 | Green: Band 2 | Blue: Band 1", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # 2. Color Infrared CIR
        axes[1].imshow(cir)
        axes[1].set_title("Color Infrared CIR (False Color)\nRed: NIR | Green: Red | Blue: Green\n[Forest = Crimson / Red, Deforestation = Cyan]", fontsize=11, fontweight="bold", color="darkred")
        axes[1].axis("off")

        # 3. SWIR Composite
        axes[2].imshow(swir)
        axes[2].set_title("SWIR Agriculture / Moisture Composite\nRed: SWIR | Green: NIR | Blue: Red\n[Forest = Lush Green, Cleared Soil = Orange/Copper]", fontsize=11, fontweight="bold", color="darkorange")
        axes[2].axis("off")

        fig.suptitle("Multispectral Band Composites for Forest & Deforestation Identification", fontsize=14, fontweight="bold", y=1.03)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f" Saved composites comparison plot to: {output_path}")
        plt.close(fig)

    def plot_spectral_signatures(self, output_path: Optional[str] = None):
        """
        Extracts sample pixels from 4 key land cover classes:
          - Dense Healthy Forest
          - Deforested / Cleared Soil
          - Water (River)
          - Access Road / Compacted Dirt
        Plots spectral reflectance profiles across Blue, Green, Red, NIR, SWIR.
        """
        # Sampling representative patches based on generated geography
        # Forest: Top-left dense area (row 40-70, col 40-70)
        forest_sample = self.bands_data[:, 40:70, 40:70]
        # Cleared Land: Inside clearing 1 (row 120-140, col 130-150)
        cleared_sample = self.bands_data[:, 120:140, 130:150]
        # Water: Meandering river (row 380-400, col 240-260)
        water_sample = self.bands_data[:, 380:400, 240:260]
        # Road: Road line (row 185-189, col 20-40)
        road_sample = self.bands_data[:, 185:189, 20:40]

        targets = {
            "Dense Forest Canopy": (forest_sample, "#1E824C", "o-"),
            "Deforested / Cleared Soil": (cleared_sample, "#D35400", "s-"),
            "Water Body (River)": (water_sample, "#2980B9", "^-"),
            "Logging Road": (road_sample, "#7F8C8D", "d-"),
        }

        wavelengths = [490, 560, 665, 842, 1610]
        band_labels = ["Blue\n(490nm)", "Green\n(560nm)", "Red\n(665nm)", "NIR\n(842nm)", "SWIR\n(1610nm)"]

        fig, ax = plt.subplots(figsize=(9, 5.5))

        for label, (sample, color, marker) in targets.items():
            means = [np.mean(sample[i]) for i in range(len(wavelengths))]
            stds = [np.std(sample[i]) for i in range(len(wavelengths))]
            ax.errorbar(
                range(len(wavelengths)), means, yerr=stds,
                label=label, color=color, fmt=marker, linewidth=2.5, markersize=8, capsize=4
            )

        ax.set_xticks(range(len(wavelengths)))
        ax.set_xticklabels(band_labels, fontsize=10, fontweight="bold")
        ax.set_ylabel("Surface Reflectance (0.0 to 1.0)", fontsize=11, fontweight="bold")
        ax.set_title("Spectral Reflectance Signatures Across Bands\nDemonstrating the Biophysical Response of Forest vs. Deforestation", fontsize=12, fontweight="bold", pad=12)
        ax.set_ylim(-0.02, 0.70)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

        # Annotations highlighting remote sensing physics
        ax.annotate(
            "Chlorophyll Absorption Dip\n(Forest absorbs Red for photosynthesis)",
            xy=(2, 0.035), xytext=(1.2, 0.20),
            arrowprops=dict(facecolor='black', arrowstyle="->", lw=1.2),
            fontsize=8.5, fontweight="bold", backgroundcolor="#ffffff"
        )
        ax.annotate(
            "The 'Red Edge' & Spongy Mesophyll Scattering\n(Forest reflects NIR intensely)",
            xy=(3, 0.52), xytext=(2.2, 0.60),
            arrowprops=dict(facecolor='green', arrowstyle="->", lw=1.2),
            fontsize=8.5, fontweight="bold", backgroundcolor="#ffffff"
        )
        ax.annotate(
            "High SWIR Reflectance in Dry Cleared Soil\n(Absorbed by water in healthy forest canopy)",
            xy=(4, 0.43), xytext=(3.1, 0.38),
            arrowprops=dict(facecolor='brown', arrowstyle="->", lw=1.2),
            fontsize=8.5, fontweight="bold", backgroundcolor="#ffffff"
        )

        plt.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f" Saved spectral signature plot to: {output_path}")
        plt.close(fig)

    def plot_histograms(self, output_path: Optional[str] = None):
        """Plots reflectance histograms for each band to illustrate dynamic range and distribution."""
        fig, axes = plt.subplots(1, len(self.band_dict), figsize=(18, 4), sharey=True, constrained_layout=True)
        colors = ["blue", "green", "red", "purple", "brown"]

        for ax, (b_name, b_data), col in zip(axes, self.band_dict.items(), colors):
            valid = b_data[b_data != self.meta.get("nodata", -9999.0)].flatten()
            ax.hist(valid, bins=50, color=col, alpha=0.75, edgecolor="black", linewidth=0.5)
            ax.set_title(b_name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Reflectance Value", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

        axes[0].set_ylabel("Pixel Count", fontsize=10, fontweight="bold")
        fig.suptitle("Pixel Value Distributions (Reflectance Histograms)", fontsize=13, fontweight="bold", y=1.04)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f" Saved band histograms plot to: {output_path}")
        plt.close(fig)

    def close(self):
        """Closes the underlying rasterio dataset."""
        if not self.dataset.closed:
            self.dataset.close()
