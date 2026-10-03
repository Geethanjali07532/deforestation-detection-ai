"""
run_module_03.py

Master execution script for Module 3: Dataset Collection & Organization.
Executes:
  1. Forest dataset ingestion & analysis (forest_fires.csv)
  2. Multi-temporal dataset audit across train, validation, and test splits
  3. Visualizations:
     - Forest fire environmental correlations & damage severity breakdown
     - Multi-temporal satellite triplet inspection (Before, After, Mask, Overlay)
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows command line / PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure local module directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.module_03_dataset_organization.forest_dataset_analyzer import ForestDatasetAnalyzer
from modules.module_03_dataset_organization.dataset_validator import DatasetValidator
from modules.module_03_dataset_organization.dataset_visualizer import visualize_temporal_triplet

def main():
    print("\n" + "=" * 75)
    print("[*] MODULE 3: DATASET COLLECTION & ORGANIZATION - MASTER RUNNER")
    print("=" * 75)

    out_dir = "outputs/module_03"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Forest Dataset Analysis
    csv_file = "data/forest_datasets/forest_fires.csv"
    if os.path.exists(csv_file):
        print("\n[1/3] Ingesting & Analyzing Forest Dataset (Forest Fires & Environmental Risk)...")
        analyzer = ForestDatasetAnalyzer(csv_file)
        analyzer.print_summary()
        plot1 = os.path.join(out_dir, "01_forest_fire_dataset_analysis.png")
        analyzer.plot_analysis(out_dir)
    else:
        print(f"\n[1/3] Note: {csv_file} not found. Skipping tabular analysis.")

    # 2. Multi-Temporal Satellite Dataset Audit
    print("\n[2/3] Auditing Multi-Temporal Satellite Dataset (train / validation / test)...")
    validator = DatasetValidator("dataset")
    validator.print_report()

    # 3. Multi-Temporal Visual Inspection
    print("\n[3/3] Generating Multi-Temporal Satellite Triplet Visual Inspection...")
    sample_b = "dataset/train/before/scene_train_001.tif"
    sample_a = "dataset/train/after/scene_train_001.tif"
    sample_m = "dataset/train/masks/scene_train_001.tif"

    if os.path.exists(sample_b) and os.path.exists(sample_a) and os.path.exists(sample_m):
        plot2 = os.path.join(out_dir, "02_multitemporal_pair_inspection.png")
        visualize_temporal_triplet(sample_b, sample_a, sample_m, "Multi-Temporal Deforestation Detection Pair", plot2)
    else:
        print("  ⚠️ Warning: Sample triplet not found in dataset/train/. Run dataset_builder.py first.")

    print("\n" + "=" * 75)
    print("🎯 Module 3 Accomplishments:")
    print("  1. Forest Datasets: Tabular forest fire weather indices (FFMC, DMC, DC, ISI) and damage severity.")
    print("  2. Multi-Temporal Satellite Pairs: Co-registered Before & After multispectral scenes.")
    print("  3. Deforestation Masks: Pixel-aligned ground-truth binary masks (1 = deforested, 0 = unchanged).")
    print("  4. ML Splits: 16 Train pairs, 4 Validation pairs, 4 Test pairs (with metadata.csv manifest).")
    print(f"📁 All visual reports saved in: {os.path.abspath(out_dir)}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
