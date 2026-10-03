"""
forest_dataset_analyzer.py

Module 3: Dataset Collection & Organization
Analyzes the Forest Fire & Burn Area Dataset (Montesinho Forest Reserve).
Connects tabular forest weather indices (FFMC, DMC, DC, ISI) and meteorology
with burned forest area and fire severity.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

class ForestDatasetAnalyzer:
    """Exploratory data analysis and feature engineering for forest fire datasets."""

    def __init__(self, csv_path: str = "data/forest_datasets/forest_fires.csv"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Forest dataset CSV not found at: {csv_path}")
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self._preprocess()

    def _preprocess(self):
        """Adds derived features for fire occurrence and severity classification."""
        # Binary fire occurrence
        self.df["fire_occurred"] = (self.df["area"] > 0).astype(int)

        # Log transform for heavily right-skewed area: ln(area + 1)
        self.df["log_area"] = np.log1p(self.df["area"])

        # Fire severity classification (Module 13 & 14 aligned)
        # 0: No fire (0 ha)
        # 1: Low burn (0 < area <= 5 ha)
        # 2: Moderate burn (5 < area <= 25 ha)
        # 3: Severe burn (> 25 ha)
        def categorize_severity(area):
            if area == 0:
                return "0: No Fire (0 ha)"
            elif area <= 5:
                return "1: Low Loss (<= 5 ha)"
            elif area <= 25:
                return "2: Moderate Loss (5-25 ha)"
            else:
                return "3: Severe Loss (> 25 ha)"

        self.df["severity_class"] = self.df["area"].apply(categorize_severity)

    def print_summary(self):
        """Prints diagnostic summary and statistical metrics."""
        print("=" * 70)
        print("🌲 FOREST DATASET SUMMARY & FIRE SEVERITY PROFILE (MODULE 3)")
        print("=" * 70)
        print(f"Dataset path       : {self.csv_path}")
        print(f"Total forest observations : {len(self.df)}")
        print(f"Features recorded  : {list(self.df.columns[:13])}")
        print(f"Missing values     : {self.df.isnull().sum().sum()}")
        print("-" * 70)

        # Class breakdown
        sev_counts = self.df["severity_class"].value_counts().sort_index()
        print("🔥 Forest Loss / Fire Severity Breakdown:")
        for sev, cnt in sev_counts.items():
            pct = (cnt / len(self.df)) * 100
            print(f"  • {sev:<28} : {cnt:>4} samples ({pct:>5.1f}%)")

        print("-" * 70)
        print("📊 Environmental Factors (Averages across Fire vs. No Fire):")
        stats = self.df.groupby("fire_occurred")[["temp", "RH", "wind", "FFMC", "DC", "ISI"]].mean()
        stats.index = ["No Fire (Unburned)", "Fire Occurred (Burned)"]
        print(stats.round(2).to_string())
        print("=" * 70)

    def plot_analysis(self, out_dir: str = "outputs/module_03"):
        """Generates comprehensive exploratory figures."""
        os.makedirs(out_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

        # 1. Monthly Distribution of Forest Fires
        month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        monthly = self.df.groupby("month")["fire_occurred"].agg(["count", "sum"]).reindex(month_order).dropna()
        axes[0, 0].bar(monthly.index, monthly["count"], color="#95A5A6", label="Total Observations", alpha=0.6)
        axes[0, 0].bar(monthly.index, monthly["sum"], color="#E74C3C", label="Fires Recorded (Loss > 0)")
        axes[0, 0].set_title("Forest Fire Incidents by Month\n(August & September Peak Season)", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Number of Events", fontsize=10)
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle=":", alpha=0.5)

        # 2. Temperature vs. Relative Humidity (RH) colored by Fire
        fire_mask = self.df["fire_occurred"] == 1
        axes[0, 1].scatter(self.df.loc[~fire_mask, "temp"], self.df.loc[~fire_mask, "RH"],
                           color="#2ECC71", alpha=0.6, label="No Fire (0 ha)", s=35)
        axes[0, 1].scatter(self.df.loc[fire_mask, "temp"], self.df.loc[fire_mask, "RH"],
                           color="#E67E22", alpha=0.8, label="Fire Occurred (> 0 ha)", s=45, edgecolors="black", linewidth=0.5)
        axes[0, 1].set_title("Temperature vs. Relative Humidity (RH)\n(High Heat + Low Humidity = Severe Fire Danger)", fontsize=11, fontweight="bold")
        axes[0, 1].set_xlabel("Temperature (°C)", fontsize=10)
        axes[0, 1].set_ylabel("Relative Humidity (%)", fontsize=10)
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle=":", alpha=0.5)

        # 3. Fire Severity Breakdown (Bar Chart)
        sev_counts = self.df["severity_class"].value_counts().sort_index()
        colors = ["#27AE60", "#F39C12", "#E67E22", "#C0392B"]
        axes[1, 0].bar([s.split(":")[0] for s in sev_counts.index], sev_counts.values, color=colors, edgecolor="black", linewidth=0.8)
        axes[1, 0].set_title("Forest Damage Severity Classes\n(Aligned with Modules 13 & 14)", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Severity Level (0=None, 1=Low, 2=Moderate, 3=Severe)", fontsize=10)
        axes[1, 0].set_ylabel("Count of Forest Patches", fontsize=10)
        axes[1, 0].grid(True, linestyle=":", alpha=0.5)
        for i, v in enumerate(sev_counts.values):
            axes[1, 0].text(i, v + 3, str(v), ha="center", fontweight="bold")

        # 4. Correlation with Burned Area (log_area)
        numeric_cols = ["temp", "RH", "wind", "rain", "FFMC", "DMC", "DC", "ISI"]
        corrs = self.df[numeric_cols].apply(lambda col: col.corr(self.df["log_area"]))
        axes[1, 1].barh(corrs.index, corrs.values, color="#3498DB", edgecolor="black", linewidth=0.6)
        axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=0.8)
        axes[1, 1].set_title("Correlation with Forest Burned Area (log_area)\n(DMC, DC, and Temperature are Positive Drivers)", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Pearson Correlation Coefficient", fontsize=10)
        axes[1, 1].grid(True, linestyle=":", alpha=0.5)

        fig.suptitle("Forest Fire & Vegetation Loss Analysis (Module 3)", fontsize=14, fontweight="bold")
        plot_path = os.path.join(out_dir, "01_forest_fire_dataset_analysis.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"💾 Saved comprehensive analysis plot to: {plot_path}")
        return plot_path

if __name__ == "__main__":
    analyzer = ForestDatasetAnalyzer()
    analyzer.print_summary()
    analyzer.plot_analysis()
