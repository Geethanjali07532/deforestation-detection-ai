"""
create_module_03_notebook.py
Generates the interactive Jupyter Notebook for Module 3.
"""

import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📂 Module 3: Dataset Collection & Organization\n",
            "\n",
            "Welcome to **Module 3** of the **Deforestation Detection from Satellite Images** project!\n",
            "\n",
            "### 🎯 Learning Objectives\n",
            "1. **Forest Datasets**: Analyze forest fire danger conditions and burn damage severity using the Forest Fires dataset.\n",
            "2. **Multi-Temporal Imagery**: Understand how Before (earlier date) and After (later date) satellite image pairs capture land-cover transitions.\n",
            "3. **Image-Label Relationships & Ground-Truth Masks**: Explore pixel-aligned binary change masks (`1 = Deforestation`, `0 = Unchanged`).\n",
            "4. **Dataset Structuring & Data Leakage Prevention**: Organize datasets into strict `train/`, `validation/`, and `test/` splits to prevent spatial leakage."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import os\n",
            "import sys\n",
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
            "from modules.module_03_dataset_organization.forest_dataset_analyzer import ForestDatasetAnalyzer\n",
            "from modules.module_03_dataset_organization.dataset_validator import DatasetValidator\n",
            "from modules.module_03_dataset_organization.dataset_visualizer import visualize_temporal_triplet\n",
            "\n",
            "%matplotlib inline\n",
            "print(\"✅ Module 3 components successfully loaded!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Forest Dataset: Fire Danger Indices & Damage Severity\n",
            "\n",
            "Deforestation is not solely caused by direct chainsaw/bulldozer logging; agricultural slash-and-burn fires and runaway forest fires are major drivers of canopy loss (explored further in Module 14).\n",
            "\n",
            "Let's load and analyze the tabular **Forest Fire & Weather Dataset** (`forest_fires.csv`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "analyzer = ForestDatasetAnalyzer(\"../data/forest_datasets/forest_fires.csv\")\n",
            "analyzer.print_summary()\n",
            "analyzer.plot_analysis(\"../outputs/module_03\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Multi-Temporal Satellite Dataset Audit\n",
            "\n",
            "Let's audit the organized multi-temporal directory structure:\n",
            "```\n",
            "dataset/\n",
            "├── train/ (before/, after/, masks/)\n",
            "├── validation/ (before/, after/, masks/)\n",
            "└── test/ (before/, after/, masks/)\n",
            "```"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "validator = DatasetValidator(\"../dataset\")\n",
            "validator.print_report()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Dataset Manifest (metadata.csv)\n",
            "\n",
            "Each multi-temporal observation is tracked with pixel dimensions, deforested pixel count, percentage, and area in hectares."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "manifest_df = pd.read_csv(\"../dataset/metadata.csv\")\n",
            "print(f\"Total multi-temporal triplets: {len(manifest_df)}\")\n",
            "manifest_df.head(10)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Visual Inspection of a Multi-Temporal Triplet\n",
            "\n",
            "Let's visualize the Before satellite image, After satellite image, Ground-Truth Deforestation mask, and the overlay."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "sample_b = \"../dataset/train/before/scene_train_001.tif\"\n",
            "sample_a = \"../dataset/train/after/scene_train_001.tif\"\n",
            "sample_m = \"../dataset/train/masks/scene_train_001.tif\"\n",
            "\n",
            "visualize_temporal_triplet(\n",
            "    sample_b,\n",
            "    sample_a,\n",
            "    sample_m,\n",
            "    sample_title=\"Train Sample 001 Multi-Temporal Triplet\",\n",
            "    output_path=\"../outputs/module_03/02_multitemporal_pair_inspection.png\"\n",
            ")\n",
            "plt.show()"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/03_dataset_collection_and_organization.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Module 3 notebook created at notebooks/03_dataset_collection_and_organization.ipynb")
