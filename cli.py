"""
cli.py
Command-Line Interface (CLI) for Operational Deforestation Detection & Geospatial Analytics.

Usage:
  python cli.py process --before data/before.tif --after data/after.tif --output-dir outputs/run1
  python cli.py batch --input-dir dataset/test --output-dir outputs/batch_run
  python cli.py dashboard --port 8501
  python cli.py info
"""

import os
import sys
import argparse
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def cmd_process(args):
    """Executes single-pair end-to-end pipeline."""
    from modules.module_19_pipeline_automation.pipeline_orchestrator import DeforestationPipeline

    print("\n" + "=" * 80)
    print("🛰️  OPERATIONAL PIPELINE: SINGLE SCENE INFERENCE & ANALYTICS")
    print("=" * 80)
    print(f"  • Before Scene : {args.before}")
    print(f"  • After Scene  : {args.after}")
    print(f"  • Output Dir   : {args.output_dir}")
    print(f"  • Threshold    : {args.threshold}")
    print(f"  • Export GIS   : {not args.no_gis}\n")

    pipeline = DeforestationPipeline(
        weights_path=args.weights,
        detection_threshold=args.threshold,
        min_patch_pixels=args.min_pixels,
        pixel_size_m=args.pixel_size
    )

    result = pipeline.process_pair(
        before_path=args.before,
        after_path=args.after,
        output_dir=args.output_dir,
        export_gis=not args.no_gis,
        save_visuals=True
    )

    print("\n" + "-" * 80)
    print(f"✅ Detection Completed in {result['execution_time_seconds']}s")
    print(f"  • Deforested Area : {result['detection_metrics']['total_deforested_ha']} ha ({result['detection_metrics']['deforestation_fraction_pct']}% of scene)")
    print(f"  • Low Severity    : {result['severity_distribution']['low_severity_ha']} ha")
    print(f"  • Mod Severity    : {result['severity_distribution']['moderate_severity_ha']} ha")
    print(f"  • High Severity   : {result['severity_distribution']['high_severity_ha']} ha")
    if result.get("gis_exports"):
        print(f"  • GIS Polygons    : {result['gis_exports'].get('polygon_count', 0)} alerts exported")
        print(f"  • GeoJSON         : {result['gis_exports'].get('geojson')}")
        print(f"  • COG GeoTIFF     : {result['gis_exports'].get('cog_geotiff')}")
    print("=" * 80 + "\n")


def cmd_batch(args):
    """Executes multi-scene batch pipeline."""
    from modules.module_19_pipeline_automation.batch_processor import BatchPipelineProcessor
    from modules.module_19_pipeline_automation.pipeline_orchestrator import DeforestationPipeline

    print("\n" + "=" * 80)
    print("📦 OPERATIONAL PIPELINE: MULTI-SCENE BATCH PROCESSING")
    print("=" * 80)
    print(f"  • Input Directory  : {args.input_dir}")
    print(f"  • Output Directory : {args.output_dir}")
    print(f"  • Max Scenes       : {args.max_scenes or 'All'}\n")

    pipeline = DeforestationPipeline(weights_path=args.weights, detection_threshold=args.threshold)
    batch_engine = BatchPipelineProcessor(pipeline=pipeline)

    report = batch_engine.process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_scenes=args.max_scenes,
        export_gis=not args.no_gis
    )

    print("\n" + "-" * 80)
    print(f"✅ Batch Run Complete: {report['scenes_processed']} scenes in {report['total_execution_time_seconds']}s")
    print(f"  • Total Deforested Area : {report['regional_totals']['total_deforested_ha']} ha")
    print(f"  • Total Alerts Exported : {report['regional_totals']['total_alerts_count']} polygons")
    print(f"  • Summary CSV           : {report['summary_csv']}")
    print(f"  • Regional Visualization: {report['regional_visualization']}")
    print("=" * 80 + "\n")


def cmd_dashboard(args):
    """Launches interactive Streamlit dashboard."""
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    if not os.path.exists(app_path):
        print(f"❌ Error: Dashboard entrypoint '{app_path}' not found.")
        sys.exit(1)

    print(f"🚀 Launching Interactive Deforestation Dashboard on port {args.port}...")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(args.port)]
    subprocess.run(cmd)


def cmd_info(args):
    """Prints system environment & module status."""
    import torch
    import rasterio
    import shapely

    print("\n" + "=" * 80)
    print("🌲 DEFORESTATION DETECTION SYSTEM: ENVIRONMENT & DIAGNOSTICS")
    print("=" * 80)
    print(f"  • Python Version      : {sys.version.split()[0]}")
    print(f"  • PyTorch Version     : {torch.__version__}")
    print(f"  • CUDA Available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  • GPU Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"  • Rasterio Version    : {rasterio.__version__}")
    print(f"  • Shapely Version     : {shapely.__version__}")
    print("-" * 80)
    print("  • Registered Project Modules:")
    for mod_num in range(2, 20):
        mod_name = f"module_{mod_num:02d}"
        mod_dir = os.path.join(os.path.dirname(__file__), "modules")
        matching = [d for d in os.listdir(mod_dir) if d.startswith(mod_name)] if os.path.exists(mod_dir) else []
        status = f"✅ ({matching[0]})" if matching else "❌ (Pending)"
        print(f"    - Module {mod_num:02d}: {status}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="🌲 Operational AI & Geospatial CLI for Satellite Deforestation Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: process
    proc_parser = subparsers.add_parser("process", help="Run end-to-end detection on a before/after image pair")
    proc_parser.add_argument("--before", "-b", required=True, help="Path to pre-disturbance GeoTIFF")
    proc_parser.add_argument("--after", "-a", required=True, help="Path to post-disturbance GeoTIFF")
    proc_parser.add_argument("--output-dir", "-o", default="outputs/cli_run", help="Output directory")
    proc_parser.add_argument("--weights", "-w", default=None, help="Optional trained model weights path")
    proc_parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Change probability threshold (default: 0.5)")
    proc_parser.add_argument("--min-pixels", type=int, default=5, help="Minimum patch size in pixels (default: 5)")
    proc_parser.add_argument("--pixel-size", type=float, default=10.0, help="Pixel resolution in meters (default: 10.0)")
    proc_parser.add_argument("--no-gis", action="store_true", help="Disable GIS vector & COG export")
    proc_parser.set_defaults(func=cmd_process)

    # Command: batch
    batch_parser = subparsers.add_parser("batch", help="Run automated batch processing over a dataset directory")
    batch_parser.add_argument("--input-dir", "-i", required=True, help="Directory containing before/ and after/ scenes")
    batch_parser.add_argument("--output-dir", "-o", default="outputs/cli_batch_run", help="Output directory")
    batch_parser.add_argument("--weights", "-w", default=None, help="Optional model weights")
    batch_parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Change threshold")
    batch_parser.add_argument("--max-scenes", "-m", type=int, default=None, help="Maximum number of scene pairs to process")
    batch_parser.add_argument("--no-gis", action="store_true", help="Disable GIS export")
    batch_parser.set_defaults(func=cmd_batch)

    # Command: dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Launch the interactive Streamlit dashboard")
    dash_parser.add_argument("--port", "-p", type=int, default=8501, help="Port to run Streamlit server (default: 8501)")
    dash_parser.set_defaults(func=cmd_dashboard)

    # Command: info
    info_parser = subparsers.add_parser("info", help="Display environment diagnostics & module inventory")
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
