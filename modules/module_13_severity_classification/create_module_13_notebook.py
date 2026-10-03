"""
create_module_13_notebook.py
Generates the interactive Jupyter Notebook for Module 13: Deforestation Severity Classification.
"""

import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🌲 Module 13: Deforestation Severity & Post-Disturbance Classification\n",
            "\n",
            "Welcome to **Module 13** of the **Deforestation Detection from Satellite Images** project!\n",
            "\n",
            "### 🎯 Learning Objectives\n",
            "1. **Disturbance Severity Quantification**: Move beyond binary change detection to assess the magnitude of forest loss.\n",
            "2. **Normalized Burn Ratio (NBR & dNBR)**: Master USGS spectral standards for canopy mortality and soil exposure.\n",
            "3. **Relativized Indices (RdNBR & RBR)**: Apply relativized damage metrics robust to pre-fire baseline vegetation density.\n",
            "4. **Multi-Class Machine Learning**: Predict 4 discrete severity tiers (Undisturbed, Low, Moderate, High) with Random Forest.\n",
            "5. **Spatial Area Accounting**: Calculate exact hectares and percentage of forest canopy affected across severity tiers."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import os\n",
            "import sys\n",
            "import glob\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import matplotlib.colors as mcolors\n",
            "import rasterio\n",
            "\n",
            "# Add project root to sys.path\n",
            "project_root = os.path.abspath('..')\n",
            "if project_root not in sys.path:\n",
            "    sys.path.insert(0, project_root)\n",
            "\n",
            "from modules.module_13_severity_classification.severity_indices import DisturbanceSeverityCalculator\n",
            "from modules.module_13_severity_classification.severity_classifier import ForestDisturbanceClassifier\n",
            "\n",
            "%matplotlib inline\n",
            "calc = DisturbanceSeverityCalculator()\n",
            "clf = ForestDisturbanceClassifier()\n",
            "print(\"✅ Module 13 Severity Environment Initialized Successfully!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Compute Continuous Disturbance Indices (dNBR, RdNBR, dNDVI)\n",
            "\n",
            "Let's load a test satellite scene pair and calculate continuous disturbance metrics."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "test_scenes_b = sorted(glob.glob(\"../dataset/test/before/*.tif\"))\n",
            "test_scenes_a = sorted(glob.glob(\"../dataset/test/after/*.tif\"))\n",
            "\n",
            "with rasterio.open(test_scenes_b[0]) as s1:\n",
            "    t1 = s1.read()\n",
            "with rasterio.open(test_scenes_a[0]) as s2:\n",
            "    t2 = s2.read()\n",
            "\n",
            "idx_dict = calc.compute_all_indices(t1, t2)\n",
            "print(f\"Computed dNBR (min: {idx_dict['dnbr'].min():.3f}, max: {idx_dict['dnbr'].max():.3f}, mean: {idx_dict['dnbr'].mean():.3f})\")\n",
            "print(f\"Computed RdNBR (min: {idx_dict['rdnbr'].min():.3f}, max: {idx_dict['rdnbr'].max():.3f})\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Visualizing Spectral Disturbance Indices\n",
            "\n",
            "We inspect continuous dNBR, RdNBR, dNDVI, and moisture deficit (dNDMI)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))\n",
            "\n",
            "im0 = axes[0].imshow(idx_dict[\"dnbr\"], cmap=\"RdYlGn_r\", vmin=-0.2, vmax=1.0)\n",
            "axes[0].set_title(\"Differenced NBR (dNBR)\", fontweight=\"bold\")\n",
            "axes[0].axis(\"off\")\n",
            "plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)\n",
            "\n",
            "im1 = axes[1].imshow(idx_dict[\"rdnbr\"], cmap=\"inferno\", vmin=0, vmax=1.5)\n",
            "axes[1].set_title(\"Relativized dNBR (RdNBR)\", fontweight=\"bold\")\n",
            "axes[1].axis(\"off\")\n",
            "plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)\n",
            "\n",
            "im2 = axes[2].imshow(idx_dict[\"dndvi\"], cmap=\"coolwarm\", vmin=-0.5, vmax=0.8)\n",
            "axes[2].set_title(\"Delta NDVI (Canopy Loss)\", fontweight=\"bold\")\n",
            "axes[2].axis(\"off\")\n",
            "plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)\n",
            "\n",
            "im3 = axes[3].imshow(idx_dict[\"dndmi\"], cmap=\"BrBG\", vmin=-0.5, vmax=0.8)\n",
            "axes[3].set_title(\"Delta NDMI (Moisture Deficit)\", fontweight=\"bold\")\n",
            "axes[3].axis(\"off\")\n",
            "plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Classify USGS Disturbance Severity Tiers & Area Breakdown\n",
            "\n",
            "Discrete classification: 0 = Undisturbed, 1 = Low, 2 = Moderate, 3 = High."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "severity_map = calc.classify_severity_usgs(idx_dict[\"dnbr\"])\n",
            "area_stats = calc.compute_severity_area_stats(severity_map, pixel_res_meters=30.0)\n",
            "\n",
            "for tier, s in area_stats.items():\n",
            "    print(f\"{tier:<26}: {s['area_hectares']:>8.2f} ha ({s['percentage']:>5.2f}% of landscape)\")"
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

out_path = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "13_deforestation_severity_classification.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Created Module 13 Notebook at: {out_path}")
