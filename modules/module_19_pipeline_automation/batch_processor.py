"""
batch_processor.py
Automated Batch Processing Engine for Processing Multi-Scene Satellite Imagery Archives.
"""

import os
import glob
import json
import time
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .pipeline_orchestrator import DeforestationPipeline


class BatchPipelineProcessor:
    """
    Automated batch processing engine for continuous satellite monitoring.
    Scans a directory of temporal scene pairs, runs the full analysis pipeline,
    and aggregates regional disturbance statistics into CSV and JSON reports.
    """

    def __init__(self, pipeline: Optional[DeforestationPipeline] = None):
        self.pipeline = pipeline if pipeline else DeforestationPipeline()

    def discover_pairs(self, base_dir: str) -> List[Dict[str, str]]:
        """
        Discovers temporal scene pairs. Supports two common structures:
        1. Subdirectories `before/` and `after/` containing matching filenames.
        2. Files matching `*_before.tif` and `*_after.tif` or `*t1*.tif` and `*t2*.tif`.
        """
        pairs = []

        before_dir = os.path.join(base_dir, "before")
        after_dir = os.path.join(base_dir, "after")

        if os.path.exists(before_dir) and os.path.exists(after_dir):
            before_files = sorted(glob.glob(os.path.join(before_dir, "*.tif")))
            for b_file in before_files:
                base_name = os.path.basename(b_file)
                a_file = os.path.join(after_dir, base_name)
                if os.path.exists(a_file):
                    pairs.append({
                        "scene_id": os.path.splitext(base_name)[0],
                        "before": b_file,
                        "after": a_file
                    })
        else:
            # Look for flat patterns
            all_tifs = sorted(glob.glob(os.path.join(base_dir, "*.tif*")))
            befores = [f for f in all_tifs if "before" in f.lower() or "_t1" in f.lower()]
            for b_file in befores:
                a_candidate = b_file.lower().replace("before", "after").replace("_t1", "_t2")
                # find real file matching candidate
                matched = [f for f in all_tifs if f.lower() == a_candidate]
                if matched:
                    pairs.append({
                        "scene_id": os.path.basename(b_file).split("_")[0],
                        "before": b_file,
                        "after": matched[0]
                    })

        return pairs

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        max_scenes: Optional[int] = None,
        export_gis: bool = True
    ) -> Dict[str, Any]:
        """
        Processes all discovered scene pairs in input_dir and saves outputs in output_dir.
        """
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        pairs = self.discover_pairs(input_dir)
        if not pairs:
            raise FileNotFoundError(f"No valid temporal scene pairs found in '{input_dir}'")

        if max_scenes:
            pairs = pairs[:max_scenes]

        print(f"📦 Batch Pipeline: Processing {len(pairs)} scene pairs...")
        results = []

        for idx, pair in enumerate(pairs, 1):
            scene_id = pair["scene_id"]
            scene_out = os.path.join(output_dir, scene_id)
            print(f"  [{idx}/{len(pairs)}] Processing Scene: {scene_id}...")

            res = self.pipeline.process_pair(
                before_path=pair["before"],
                after_path=pair["after"],
                output_dir=scene_out,
                export_gis=export_gis,
                save_visuals=True
            )
            res["scene_id"] = scene_id
            results.append(res)

        # Aggregate Statistics across scenes
        total_deforested_ha = sum(r["detection_metrics"]["total_deforested_ha"] for r in results)
        total_low_sev_ha = sum(r["severity_distribution"]["low_severity_ha"] for r in results)
        total_mod_sev_ha = sum(r["severity_distribution"]["moderate_severity_ha"] for r in results)
        total_high_sev_ha = sum(r["severity_distribution"]["high_severity_ha"] for r in results)
        total_polygons = sum(r["gis_exports"].get("polygon_count", 0) for r in results)

        # Aggregate Drivers
        driver_aggregate = {}
        for r in results:
            for driver, cnt in r["driver_attribution"].items():
                driver_aggregate[driver] = driver_aggregate.get(driver, 0) + cnt

        # Create Summary DataFrame
        summary_rows = []
        for r in results:
            summary_rows.append({
                "scene_id": r["scene_id"],
                "deforested_ha": r["detection_metrics"]["total_deforested_ha"],
                "deforested_pct": r["detection_metrics"]["deforestation_fraction_pct"],
                "polygons": r["gis_exports"].get("polygon_count", 0),
                "low_severity_ha": r["severity_distribution"]["low_severity_ha"],
                "mod_severity_ha": r["severity_distribution"]["moderate_severity_ha"],
                "high_severity_ha": r["severity_distribution"]["high_severity_ha"],
                "execution_seconds": r["execution_time_seconds"]
            })
        df_summary = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "batch_regional_summary.csv")
        df_summary.to_csv(csv_path, index=False)

        # Generate Aggregate Visualization
        viz_path = os.path.join(output_dir, "batch_regional_overview.png")
        self._plot_batch_summary(df_summary, driver_aggregate, viz_path)

        batch_report = {
            "status": "SUCCESS",
            "scenes_processed": len(results),
            "total_execution_time_seconds": round(time.time() - start_time, 2),
            "regional_totals": {
                "total_deforested_ha": round(total_deforested_ha, 2),
                "total_alerts_count": int(total_polygons),
                "severity_breakdown_ha": {
                    "low": round(total_low_sev_ha, 2),
                    "moderate": round(total_mod_sev_ha, 2),
                    "high": round(total_high_sev_ha, 2)
                },
                "driver_breakdown": driver_aggregate
            },
            "summary_csv": csv_path,
            "regional_visualization": viz_path,
            "scene_details": results
        }

        report_json = os.path.join(output_dir, "batch_regional_summary.json")
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(batch_report, f, indent=2)

        print(f"✅ Batch Processing Completed in {batch_report['total_execution_time_seconds']}s")
        print(f"📊 Regional Deforested Area: {total_deforested_ha:.2f} ha across {total_polygons} alert polygons")
        return batch_report

    def _plot_batch_summary(self, df: pd.DataFrame, drivers: Dict[str, int], output_path: str):
        """Plots regional deforestation breakdown across scenes and attributed drivers."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Deforestation per scene
        axes[0].bar(df["scene_id"], df["deforested_ha"], color="#d7191c", edgecolor="black", alpha=0.85)
        axes[0].set_title("Deforestation Area by Scene (Hectares)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Hectares (ha)")
        axes[0].set_xticklabels(df["scene_id"], rotation=45, ha="right")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        # 2. Driver Distribution
        if drivers:
            labels = list(drivers.keys())
            counts = list(drivers.values())
            colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3"][:len(labels)]
            axes[1].pie(counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140,
                        wedgeprops=dict(edgecolor="black", linewidth=1.2))
            axes[1].set_title("Attributed Disturbance Drivers", fontsize=11, fontweight="bold")
        else:
            axes[1].text(0.5, 0.5, "No disturbance drivers detected", ha="center", va="center")

        plt.suptitle("Batch Regional Deforestation Summary Dashboard", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
