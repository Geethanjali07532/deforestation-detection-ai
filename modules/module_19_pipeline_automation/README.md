# Module 19: End-to-End Pipeline Automation & CLI Deployment

## 📌 Overview
Module 19 bridges research models and operational production by delivering:
1. **`DeforestationPipeline`**: A unified, automated orchestrator that accepts bi-temporal satellite GeoTIFFs and seamlessly performs:
   - Radiometric normalization and multi-band alignment
   - Deep Siamese change detection inference
   - USGS 4-tier disturbance severity classification (dNBR, RdNBR)
   - Landscape ecology patch morphology and driver attribution (Wildfire, Road Encroachment, Selective Logging, Clearcut)
   - Georeferenced topological polygonization and Douglas-Peucker simplification
   - Comprehensive GIS export (RFC 7946 GeoJSON, ESRI Shapefiles, Cloud-Optimized GeoTIFFs)
2. **`BatchPipelineProcessor`**: Multi-scene automated processing engine that scans image archives, executes end-to-end analytics on all temporal pairs, and aggregates regional disturbance statistics into CSV, JSON, and visual dashboards.
3. **`cli.py`**: Command-line tool for field deployment and unattended server execution.

---

## 💻 CLI Usage Guide

### 1. Process Single Scene Pair
```bash
python cli.py process \
  --before dataset/test/before/scene_test_001.tif \
  --after dataset/test/after/scene_test_001.tif \
  --output-dir outputs/operational_run_01 \
  --threshold 0.5 \
  --pixel-size 10.0
```

### 2. Multi-Scene Batch Processing
```bash
python cli.py batch \
  --input-dir dataset/test \
  --output-dir outputs/batch_regional_run \
  --max-scenes 10
```

### 3. Launch Web Dashboard
```bash
python cli.py dashboard --port 8501
```

### 4. System Diagnostics
```bash
python cli.py info
```

---

## 📊 Operational Validation
- **Single Scene Runtime:** 5.55 seconds (CPU)
- **Deforested Area Detected:** 81.54 ha across 32 topological polygons
- **Batch Processing Rate:** 4 scenes in 19.71s (4.93s / scene)
- **Regional Alerts:** 103 vector polygons exported across 273.38 ha
- **Attributed Drivers:**
  - Road Encroachments: 14 patches
  - Agricultural Clearcuts: 6 patches
  - Selective Logging Gaps: 6 patches
  - Wildfire Scars: 3 patches

---

## 📁 Artifacts Produced
- `cli.py`: Production-grade command-line interface.
- `outputs/module_19/single_scene_run/pipeline_summary_composite.png`: 5-panel operational summary chart.
- `outputs/module_19/single_scene_run/deforestation_alerts.geojson`: Vectorized alerts with severity and geometry attributes.
- `outputs/module_19/single_scene_run/deforestation_severity_cog.tif`: Cloud-Optimized GeoTIFF with 128x128 tiles.
- `outputs/module_19/batch_run/batch_regional_overview.png`: Regional loss bar chart & driver pie chart.
- `outputs/module_19/batch_run/batch_regional_summary.csv`: Aggregated scene-by-scene tabular report.
- `outputs/module_19/module_19_pipeline_audit.json`: Complete audit log for operational compliance.
- `notebooks/19_pipeline_automation_and_cli.ipynb`: Interactive Jupyter analysis notebook.
