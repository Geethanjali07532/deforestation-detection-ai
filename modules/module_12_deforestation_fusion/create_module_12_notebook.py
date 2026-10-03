"""
create_module_12_notebook.py
Generates the interactive Jupyter Notebook for Module 12: Deep Learning Fusion Strategies.
"""

import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🌲 Module 12: Deep Learning Fusion Strategies for Deforestation Detection\n",
            "\n",
            "Welcome to **Module 12** of the **Deforestation Detection from Satellite Images** project!\n",
            "\n",
            "### 🎯 Learning Objectives\n",
            "1. **Information Fusion Paradigms**: Understand where and how multi-temporal satellite data should be merged.\n",
            "2. **Early Fusion (EF)**: 10-band input stacking with joint early feature extraction.\n",
            "3. **Late Fusion (LF)**: Dual unshared encoders with bottleneck concatenation.\n",
            "4. **Siamese Differencing vs Concatenation**: Compare differential vs concatenated skip pathways in weight-sharing networks.\n",
            "5. **Benchmarking & Error Decompositions**: Quantify True Positives, False Alarms (FP), and Misses (FN)."
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
            "import torch\n",
            "\n",
            "# Add project root to sys.path\n",
            "project_root = os.path.abspath('..')\n",
            "if project_root not in sys.path:\n",
            "    sys.path.insert(0, project_root)\n",
            "\n",
            "from modules.module_11_siamese_change_detection.siamese_dataset import create_siamese_dataloaders\n",
            "from modules.module_12_deforestation_fusion.fusion_models import (\n",
            "    EarlyFusionUNet, LateFusionUNet, SiameseDiffUNet, SiameseConcatUNet\n",
            ")\n",
            "from modules.module_12_deforestation_fusion.fusion_comparator import (\n",
            "    compare_fusion_strategies, count_parameters\n",
            ")\n",
            "\n",
            "%matplotlib inline\n",
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "print(f\"✅ Module 12 Fusion environment ready on device: [{device}]\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Bi-Temporal Paired DataLoaders\n",
            "\n",
            "We load paired satellite imagery $(T_1, T_2)$ and ground-truth deforestation masks."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "train_loader, val_loader, test_loader = create_siamese_dataloaders(\n",
            "    dataset_root=\"../dataset\",\n",
            "    tile_size=128,\n",
            "    batch_size=8,\n",
            "    max_train_tiles=32,\n",
            "    max_val_tiles=8\n",
            ")\n",
            "\n",
            "print(f\"Train pairs: {len(train_loader.dataset)} | Val pairs: {len(val_loader.dataset)} | Test pairs: {len(test_loader.dataset)}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Inspect the 4 Fusion Architectures\n",
            "\n",
            "Let's compare the parameter counts across Early Fusion, Late Fusion, Siamese Differencing, and Siamese Concatenation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "models_to_test = {\n",
            "    \"Early Fusion (EF)\": EarlyFusionUNet(in_channels_per_image=5, out_channels=1, base_features=16),\n",
            "    \"Late Fusion (LF)\": LateFusionUNet(in_channels_per_image=5, out_channels=1, base_features=16),\n",
            "    \"Siamese Differencing\": SiameseDiffUNet(in_channels_per_image=5, out_channels=1, base_features=16),\n",
            "    \"Siamese Concatenation\": SiameseConcatUNet(in_channels_per_image=5, out_channels=1, base_features=16),\n",
            "}\n",
            "\n",
            "for name, m in models_to_test.items():\n",
            "    print(f\"{name:<24}: {count_parameters(m):>9,} parameters\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Train & Benchmark All Fusion Strategies\n",
            "\n",
            "We train each model under identical conditions (Compound Change Loss, Adam optimizer, lr=1e-3)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "leaderboard_df, trained_models = compare_fusion_strategies(\n",
            "    models_dict=models_to_test,\n",
            "    train_loader=train_loader,\n",
            "    val_loader=val_loader,\n",
            "    test_loader=test_loader,\n",
            "    epochs=4,\n",
            "    lr=1e-3,\n",
            "    device=device\n",
            ")\n",
            "\n",
            "leaderboard_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Visualizing Performance Trade-offs\n",
            "\n",
            "Let's compare Mean IoU, Dice Score, CPU Latency, and Parameter efficiency."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))\n",
            "\n",
            "archs = leaderboard_df[\"Fusion Strategy\"].tolist()\n",
            "ious = [v * 100 for v in leaderboard_df[\"Mean IoU\"]]\n",
            "dices = [v * 100 for v in leaderboard_df[\"Dice Score\"]]\n",
            "latencies = leaderboard_df[\"Latency (ms)\"].tolist()\n",
            "params_k = [int(p.replace(\",\", \"\")) / 1000.0 for p in leaderboard_df[\"Parameters\"]]\n",
            "\n",
            "x = np.arange(len(archs))\n",
            "width = 0.35\n",
            "ax1.bar(x - width/2, ious, width, label=\"Mean IoU (%)\", color=\"#2e7d32\")\n",
            "ax1.bar(x + width/2, dices, width, label=\"Dice (%)\", color=\"#0288d1\")\n",
            "ax1.set_title(\"Mean IoU & Dice Score\", fontweight=\"bold\")\n",
            "ax1.set_xticks(x)\n",
            "ax1.set_xticklabels(archs, rotation=15, ha=\"right\")\n",
            "ax1.set_ylim(0, 110)\n",
            "ax1.legend()\n",
            "ax1.grid(axis=\"y\", linestyle=\"--\", alpha=0.5)\n",
            "\n",
            "ax2.bar(archs, latencies, color=\"#ff8f00\", width=0.5)\n",
            "ax2.set_title(\"Latency per Tile (ms)\", fontweight=\"bold\")\n",
            "ax2.set_xticklabels(archs, rotation=15, ha=\"right\")\n",
            "ax2.grid(axis=\"y\", linestyle=\"--\", alpha=0.5)\n",
            "\n",
            "ax3.scatter(params_k, ious, s=200, color=\"#d81b60\", edgecolors=\"black\")\n",
            "for i, arch in enumerate(archs):\n",
            "    ax3.annotate(f\" {arch}\", (params_k[i], ious[i]), fontsize=9, fontweight=\"bold\")\n",
            "ax3.set_xlabel(\"Parameters (Thousands)\", fontweight=\"bold\")\n",
            "ax3.set_ylabel(\"Mean IoU (%)\", fontweight=\"bold\")\n",
            "ax3.set_title(\"Parameters vs Mean IoU\", fontweight=\"bold\")\n",
            "ax3.grid(True, linestyle=\"--\", alpha=0.5)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
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

out_path = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "12_deep_learning_deforestation_fusion.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Created Module 12 Notebook at: {out_path}")
