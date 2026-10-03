"""
pipeline_orchestrator.py
End-to-End Automated Pipeline Orchestrator for Deforestation Detection, Severity, Driver Attribution, and GIS Vector Export.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Internal Module Imports
from modules.module_11_siamese_change_detection.siamese_model import SiameseUNetChangeDetector
from modules.module_13_severity_classification.severity_indices import DisturbanceSeverityCalculator
from modules.module_13_severity_classification.severity_classifier import ForestDisturbanceClassifier
from modules.module_14_pattern_analysis.morphological_analyzer import PatchMorphologyExtractor
from modules.module_14_pattern_analysis.driver_classifier import DisturbanceDriverClassifier
from modules.module_16_geospatial_export.vectorizer import RasterPolygonizer
from modules.module_16_geospatial_export.gis_exporter import export_geojson, export_shapefile, export_cog_geotiff


class DeforestationPipeline:
    """
    Automated end-to-end inference & analytics pipeline.
    Chains ingestion -> deep Siamese detection -> severity classification -> driver attribution -> GIS vector export.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        detection_threshold: float = 0.5,
        min_patch_pixels: int = 5,
        pixel_size_m: float = 10.0
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.detection_threshold = detection_threshold
        self.min_patch_pixels = min_patch_pixels
        self.pixel_size_m = pixel_size_m

        # 1. Initialize Siamese Network
        self.model = SiameseUNetChangeDetector(in_channels=5, base_features=16).to(self.device)
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
        else:
            self.model.eval()

        # 2. Analytics Engines
        self.severity_calc = DisturbanceSeverityCalculator()
        self.severity_classifier = ForestDisturbanceClassifier()
        self.morph_extractor = PatchMorphologyExtractor(min_patch_pixels=self.min_patch_pixels, pixel_res_meters=self.pixel_size_m)
        self.driver_classifier = DisturbanceDriverClassifier()
        self.vectorizer = RasterPolygonizer(min_area_ha=(self.min_patch_pixels * self.pixel_size_m**2) / 10000.0, pixel_res_meters=self.pixel_size_m)

    def load_raster(self, raster_path: str) -> Tuple[np.ndarray, Affine, str]:
        """
        Reads a multi-band GeoTIFF and returns (bands [C, H, W], affine_transform, crs_string).
        """
        with rasterio.open(raster_path) as src:
            data = src.read().astype(np.float32)
            transform = src.transform
            crs = src.crs.to_string() if src.crs else "EPSG:32621"

        # Normalize to [0, 1] if in [0, 10000] reflectance
        if np.nanmax(data) > 10.0:
            data = np.clip(data / 10000.0, 0.0, 1.0)
        return data, transform, crs

    def process_pair(
        self,
        before_path: str,
        after_path: str,
        output_dir: str,
        export_gis: bool = True,
        save_visuals: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the complete operational pipeline on a before/after satellite pair.
        """
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        # 1. Ingest imagery
        t1_data, transform, crs = self.load_raster(before_path)
        t2_data, _, _ = self.load_raster(after_path)

        # Ensure 5 channels (B, G, R, NIR, SWIR)
        if t1_data.shape[0] < 5 or t2_data.shape[0] < 5:
            raise ValueError(f"Expected at least 5 bands, got {t1_data.shape[0]} and {t2_data.shape[0]}")

        C, H, W = t1_data.shape

        # 2. Deep Siamese Change Detection
        t1_tensor = torch.from_numpy(t1_data[:5]).unsqueeze(0).to(self.device)
        t2_tensor = torch.from_numpy(t2_data[:5]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(t1_tensor, t2_tensor)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Hybrid detection: refine with delta-NDVI if available
        # Band mapping: 0=Blue, 1=Green, 2=Red, 3=NIR, 4=SWIR
        nir1, red1 = t1_data[3], t1_data[2]
        nir2, red2 = t2_data[3], t2_data[2]
        ndvi1 = (nir1 - red1) / (nir1 + red1 + 1e-6)
        ndvi2 = (nir2 - red2) / (nir2 + red2 + 1e-6)
        d_ndvi = ndvi1 - ndvi2

        # Combine deep model probability with spectral loss
        spectral_change = (d_ndvi > 0.25).astype(np.float32)
        combined_prob = np.clip(0.6 * prob_map + 0.4 * spectral_change, 0.0, 1.0)
        change_mask = (combined_prob >= self.detection_threshold).astype(np.uint8)

        # 3. Severity Classification (Module 13)
        indices = self.severity_calc.compute_all_indices(t1_data, t2_data)
        dnbr = indices["dnbr"]
        rdnbr = indices["rdnbr"]

        # Standard USGS scientific burn severity tiers
        severity_map = np.zeros_like(change_mask, dtype=np.uint8)
        severity_map[(change_mask == 1) & (dnbr < 0.27)] = 1
        severity_map[(change_mask == 1) & (dnbr >= 0.27) & (dnbr < 0.66)] = 2
        severity_map[(change_mask == 1) & (dnbr >= 0.66)] = 3

        # 4. Patch Morphology & Driver Attribution (Module 14)
        patches_list, labeled_mask = self.morph_extractor.extract_patch_metrics(change_mask, dnbr_raster=dnbr)
        driver_counts = {}
        if patches_list:
            df_patches = self.driver_classifier.classify_all_patches(patches_list)
            driver_counts = df_patches["driver"].value_counts().to_dict()
            total_cleared_ha = float(df_patches["area_hectares"].sum())
        else:
            total_cleared_ha = 0.0

        # Severity area breakdown
        pixel_area_ha = (self.pixel_size_m * self.pixel_size_m) / 10000.0
        severity_audit = {
            "undisturbed_ha": float(np.sum(severity_map == 0) * pixel_area_ha),
            "low_severity_ha": float(np.sum(severity_map == 1) * pixel_area_ha),
            "moderate_severity_ha": float(np.sum(severity_map == 2) * pixel_area_ha),
            "high_severity_ha": float(np.sum(severity_map == 3) * pixel_area_ha),
            "total_deforested_ha": float(np.sum(change_mask == 1) * pixel_area_ha)
        }

        # 5. Vectorization & GIS Export (Module 16)
        gis_exports = {}
        if export_gis:
            features = self.vectorizer.polygonize(
                change_mask,
                transform=transform,
                crs=crs,
                severity_map=severity_map,
                dnbr_raster=dnbr
            )
            geojson_path = os.path.join(output_dir, "deforestation_alerts.geojson")
            shapefile_path = os.path.join(output_dir, "shapefile", "deforestation_alerts.shp")
            cog_path = os.path.join(output_dir, "deforestation_severity_cog.tif")

            export_geojson(features, geojson_path)
            export_shapefile(features, shapefile_path)
            export_cog_geotiff(severity_map, cog_path, transform=transform, crs=crs)

            gis_exports = {
                "geojson": geojson_path,
                "shapefile": shapefile_path,
                "cog_geotiff": cog_path,
                "polygon_count": len(features)
            }

        # 6. Save Pipeline Result Composite Plot
        viz_path = None
        if save_visuals:
            viz_path = os.path.join(output_dir, "pipeline_summary_composite.png")
            self._render_pipeline_composite(
                t1_data, t2_data, combined_prob, change_mask, severity_map,
                output_path=viz_path, title=os.path.basename(before_path)
            )

        elapsed = time.time() - start_time

        # Compile final operational manifest
        summary_result = {
            "status": "SUCCESS",
            "execution_time_seconds": round(elapsed, 2),
            "input_scenes": {
                "before": before_path,
                "after": after_path,
                "dimensions": [int(C), int(H), int(W)],
                "crs": crs
            },
            "detection_metrics": {
                "threshold": self.detection_threshold,
                "deforested_pixels": int(np.sum(change_mask == 1)),
                "total_pixels": int(H * W),
                "deforestation_fraction_pct": round(float(np.mean(change_mask) * 100), 2),
                "total_deforested_ha": round(severity_audit["total_deforested_ha"], 2)
            },
            "severity_distribution": severity_audit,
            "driver_attribution": driver_counts,
            "gis_exports": gis_exports,
            "visualization": viz_path
        }

        # Write manifest JSON
        manifest_path = os.path.join(output_dir, "pipeline_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(summary_result, f, indent=2)

        return summary_result

    def _render_pipeline_composite(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        prob: np.ndarray,
        mask: np.ndarray,
        severity: np.ndarray,
        output_path: str,
        title: str
    ):
        """Creates a high-resolution 5-panel operational summary chart."""
        fig, axes = plt.subplots(1, 5, figsize=(22, 5))

        def to_rgb(data):
            # True Color: R=2, G=1, B=0
            rgb = np.stack([data[2], data[1], data[0]], axis=-1)
            p2, p98 = np.percentile(rgb, (2, 98))
            return np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

        axes[0].imshow(to_rgb(t1))
        axes[0].set_title("Pre-Disturbance (T1 True Color)", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(to_rgb(t2))
        axes[1].set_title("Post-Disturbance (T2 True Color)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        im_prob = axes[2].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[2].set_title("Deep Change Probability", fontsize=11, fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im_prob, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(to_rgb(t2))
        axes[3].imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.6)
        axes[3].set_title("Detected Canopy Loss (Alerts)", fontsize=11, fontweight="bold")
        axes[3].axis("off")

        # 4-Tier Severity colormap: 0=None, 1=Yellow, 2=Orange, 3=Red
        cmap_sev = matplotlib.colors.ListedColormap(["#2b83ba", "#ffffbf", "#fdae61", "#d7191c"])
        im_sev = axes[4].imshow(severity, cmap=cmap_sev, vmin=0, vmax=3)
        axes[4].set_title("USGS 4-Tier Severity", fontsize=11, fontweight="bold")
        axes[4].axis("off")
        cbar = plt.colorbar(im_sev, ax=axes[4], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(["Intact", "Low", "Mod", "High"])

        plt.suptitle(f"Operational Deforestation Pipeline — Scene: {title}", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
