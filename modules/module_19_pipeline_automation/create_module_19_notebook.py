"""
create_module_19_notebook.py
Generates the interactive Jupyter Notebook for Module 19: End-to-End Pipeline Automation & CLI Deployment.
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🛰️ Module 19: End-to-End Pipeline Automation & CLI Deployment\n",
            "\n",
            "Welcome to **Module 19** of the **Deforestation Detection from Satellite Images** project!\n",
            "\n",
            "### 🎯 Learning Objectives\n",
            "1. **Full Pipeline Orchestration**: Chain radiometric ingestion, deep Siamese inference, severity classification, driver attribution, and GIS polygonization in a single call.\n",
            "2. **Multi-Scene Batch Processing Engine**: Automatically scan and process large satellite imagery archives with regional aggregation.\n",
            "3. **Standard Geospatial Outputs**: Generate RFC 7946 GeoJSON, ESRI Shapefile bundles, and Cloud-Optimized GeoTIFFs (COG).\n",
            "4. **Command-Line Interface (CLI)**: Drive operational workflows via `python cli.py process`, `batch`, `info`, and `dashboard`."
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
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
            "\n",
            "# Add project root to sys.path\n",
            "project_root = os.path.abspath('..')\n",
            "if project_root not in sys.path:\n",
            "    sys.path.insert(0, project_root)\n",
            "\n",
            "from modules.module_19_pipeline_automation.pipeline_orchestrator import DeforestationPipeline\n",
            "from modules.module_19_pipeline_automation.batch_processor import BatchPipelineProcessor\n",
            "\n",
            "%matplotlib inline\n",
            "print(\"✅ Pipeline Automation Module Imported Successfully!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Single-Scene Pipeline Demonstration\n",
            "\n",
            "Let's inspect the 5-panel operational summary chart generated for a single satellite scene pair."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "img_path = '../outputs/module_19/single_scene_run/pipeline_summary_composite.png'\n",
            "if os.path.exists(img_path):\n",
            "    plt.figure(figsize=(18, 5))\n",
            "    plt.imshow(Image.open(img_path))\n",
            "    plt.axis('off')\n",
            "    plt.title(\"End-to-End Pipeline Inference Composite\", fontsize=13, fontweight='bold')\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"Run run_module_19.py to generate composite figure.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Multi-Scene Batch Processing & Regional Aggregation\n",
            "\n",
            "Reviewing regional statistics across the processed satellite scenes."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "csv_path = '../outputs/module_19/batch_run/batch_regional_summary.csv'\n",
            "if os.path.exists(csv_path):\n",
            "    df = pd.read_csv(csv_path)\n",
            "    display(df)\n",
            "    \n",
            "    img_batch = '../outputs/module_19/batch_run/batch_regional_overview.png'\n",
            "    if os.path.exists(img_batch):\n",
            "        plt.figure(figsize=(14, 5))\n",
            "        plt.imshow(Image.open(img_batch))\n",
            "        plt.axis('off')\n",
            "        plt.title(\"Batch Regional Deforestation Dashboard\", fontsize=13, fontweight='bold')\n",
            "        plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. CLI Command-Line Reference\n",
            "\n",
            "```bash\n",
            "# 1. Process single satellite pair\n",
            "python cli.py process --before dataset/test/before/scene_001.tif --after dataset/test/after/scene_001.tif --output-dir outputs/run_01\n",
            "\n",
            "# 2. Run batch monitoring over whole archive\n",
            "python cli.py batch --input-dir dataset/test --output-dir outputs/regional_monitoring\n",
            "\n",
            "# 3. Launch interactive web dashboard\n",
            "python cli.py dashboard --port 8501\n",
            "\n",
            "# 4. Display environment and module diagnostics\n",
            "python cli.py info\n",
            "```"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "notebooks", "19_pipeline_automation_and_cli.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"✅ Generated Notebook: {out_path}")
