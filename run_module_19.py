"""
run_module_19.py
Master runner for Module 19: End-to-End Pipeline Automation & CLI Deployment.
Validates single-pair operational inference, multi-scene batch aggregation, and GIS export.
"""

import os
import sys
import json
import glob
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modules.module_19_pipeline_automation.pipeline_orchestrator import DeforestationPipeline
from modules.module_19_pipeline_automation.batch_processor import BatchPipelineProcessor


def main():
    print("\n" + "=" * 80)
    print("🛰️  MODULE 19: END-TO-END PIPELINE AUTOMATION & CLI DEPLOYMENT")
    print("=" * 80)

    output_root = os.path.join(os.path.dirname(__file__), "outputs", "module_19")
    os.makedirs(output_root, exist_ok=True)

    # 1. Discover sample test pair
    test_dir = os.path.join(os.path.dirname(__file__), "dataset", "test")
    before_dir = os.path.join(test_dir, "before")
    after_dir = os.path.join(test_dir, "after")

    before_scenes = sorted(glob.glob(os.path.join(before_dir, "*.tif")))
    if not before_scenes:
        print(f"❌ Error: No scenes found in {before_dir}")
        sys.exit(1)

    sample_before = before_scenes[0]
    sample_after = os.path.join(after_dir, os.path.basename(sample_before))
    scene_id = os.path.splitext(os.path.basename(sample_before))[0]

    print(f"\n[1/3] Executing Single-Scene End-to-End Operational Pipeline ({scene_id})...")
    single_out = os.path.join(output_root, "single_scene_run")

    pipeline = DeforestationPipeline(detection_threshold=0.5, pixel_size_m=10.0)
    single_result = pipeline.process_pair(
        before_path=sample_before,
        after_path=sample_after,
        output_dir=single_out,
        export_gis=True,
        save_visuals=True
    )

    print(f"  • Scene Processed in : {single_result['execution_time_seconds']}s")
    print(f"  • Deforested Area    : {single_result['detection_metrics']['total_deforested_ha']} ha")
    print(f"  • Polygons Vectorized: {single_result['gis_exports']['polygon_count']}")
    print(f"  • GeoJSON Generated  : {os.path.basename(single_result['gis_exports']['geojson'])}")
    print(f"  • COG GeoTIFF Export : {os.path.basename(single_result['gis_exports']['cog_geotiff'])}")

    # 2. Executing Batch Processing on Test Directory
    print(f"\n[2/3] Executing Automated Batch Pipeline on Test Directory...")
    batch_out = os.path.join(output_root, "batch_run")
    batch_engine = BatchPipelineProcessor(pipeline=pipeline)

    batch_report = batch_engine.process_batch(
        input_dir=test_dir,
        output_dir=batch_out,
        max_scenes=4,
        export_gis=True
    )

    print(f"  • Batch Execution Time   : {batch_report['total_execution_time_seconds']}s")
    print(f"  • Scenes Analyzed        : {batch_report['scenes_processed']}")
    print(f"  • Total Regional Loss    : {batch_report['regional_totals']['total_deforested_ha']} ha")
    print(f"  • Total Alert Polygons   : {batch_report['regional_totals']['total_alerts_count']}")
    print(f"  • Disturbance Drivers    : {batch_report['regional_totals']['driver_breakdown']}")

    # 3. Assertions & Validation
    print(f"\n[3/3] Validating Output Deliverables & Compliance...")
    assert os.path.exists(single_result["gis_exports"]["geojson"]), "GeoJSON not found"
    assert os.path.exists(single_result["gis_exports"]["cog_geotiff"]), "COG GeoTIFF not found"
    assert os.path.exists(batch_report["summary_csv"]), "Batch CSV not found"
    assert os.path.exists(batch_report["regional_visualization"]), "Regional overview figure not found"

    # Save summary manifest
    manifest_path = os.path.join(output_root, "module_19_pipeline_audit.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "single_scene": single_result,
            "batch_processing": {
                "scenes_processed": batch_report["scenes_processed"],
                "total_time_seconds": batch_report["total_execution_time_seconds"],
                "regional_loss_ha": batch_report["regional_totals"]["total_deforested_ha"],
                "total_alerts": batch_report["regional_totals"]["total_alerts_count"],
                "driver_breakdown": batch_report["regional_totals"]["driver_breakdown"],
                "summary_csv": batch_report["summary_csv"]
            }
        }, f, indent=2)

    print(f"  • Master audit manifest saved -> '{manifest_path}'")
    print("\n" + "=" * 80)
    print("✅ MODULE 19: PIPELINE AUTOMATION & CLI DEPLOYMENT SUCCESSFUL!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
