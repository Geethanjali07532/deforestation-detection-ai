"""
create_module_16_notebook.py
Generates the interactive Jupyter Notebook for Module 16: Geospatial Vector & COG Export.
"""

import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🌲 Module 16: Geospatial Vector Polygon Extraction & GIS Export\n",
            "\n",
            "Welcome to **Module 16** of the **Deforestation Detection from Satellite Images** project!\n",
            "\n",
            "### 🎯 Learning Objectives\n",
            "1. **Operational GIS Integration**: Bridge deep learning raster predictions with vector GIS formats.\n",
            "2. **Raster-to-Vector Polygonization**: Use affine transformation matrices to produce georeferenced polygons.\n",
            "3. **Douglas-Peucker Simplification**: Eliminate pixel staircase artifacts and optimize polygon storage.\n",
            "4. **Multi-Format Export**: Generate GeoJSON FeatureCollections, ESRI Shapefiles, and Cloud-Optimized GeoTIFFs (COG).\n",
            "5. **Field Attribution**: Attach critical law enforcement attributes (Alert ID, Area, Severity, Driver, Centroid Coordinates)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import os\n",
            "import sys\n",
            "import json\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import rasterio\n",
            "\n",
            "# Add project root to sys.path\n",
            "project_root = os.path.abspath('..')\n",
            "if project_root not in sys.path:\n",
            "    sys.path.insert(0, project_root)\n",
            "\n",
            "from modules.module_16_geospatial_export.vectorizer import RasterPolygonizer\n",
            "from modules.module_16_geospatial_export.gis_exporter import export_geojson, export_shapefile, export_cog_geotiff\n",
            "\n",
            "%matplotlib inline\n",
            "print(\"✅ Module 16 GIS Export Environment Ready!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Satellite Scene & Georeferencing Metadata\n",
            "\n",
            "Inspect affine transform, coordinate reference system (CRS), and raster masks."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "with rasterio.open(\"../dataset/test/masks/scene_test_001.tif\") as sm:\n",
            "    mask = sm.read(1)\n",
            "    transform = sm.transform\n",
            "    crs = sm.crs\n",
            "\n",
            "print(f\"CRS: {crs}\")\n",
            "print(f\"Transform: {transform}\")\n",
            "print(f\"Mask shape: {mask.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Vectorize Raster to GeoJSON Features\n",
            "\n",
            "Apply Douglas-Peucker simplification and extract attribute metadata."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "vectorizer = RasterPolygonizer(min_area_ha=0.05, simplify_tolerance=0.0001, pixel_res_meters=30.0)\n",
            "features = vectorizer.polygonize(mask, transform=transform, crs=str(crs))\n",
            "print(f\"Extracted {len(features)} valid vector polygons\")\n",
            "\n",
            "# Preview first polygon\n",
            "features[0][\"properties\"]"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Export to GeoJSON, Shapefile, and COG GeoTIFF\n",
            "\n",
            "Export all GIS formats ready for QGIS, ArcGIS, and Web Map integration."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "out_geojson = \"../outputs/module_16/deforestation_alerts.geojson\"\n",
            "export_geojson(features, out_geojson)\n",
            "\n",
            "out_shp = \"../outputs/module_16/shapefile/deforestation_alerts.shp\"\n",
            "export_shapefile(features, out_shp)\n",
            "\n",
            "out_cog = \"../outputs/module_16/deforestation_severity_cog.tif\"\n",
            "export_cog_geotiff(mask, out_cog, transform=transform, crs=str(crs))\n",
            "\n",
            "print(\"✅ Successfully exported GeoJSON, ESRI Shapefile, and Cloud-Optimized GeoTIFF!\")"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.13"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "16_geospatial_vector_and_cog_export.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Created Module 16 Notebook at: {out_path}")
