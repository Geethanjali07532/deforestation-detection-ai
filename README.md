# Deforestation Detection from Satellite Images

Operational Satellite AI & Multi-Temporal Earth Observation Platform for Automated Canopy Disturbance Monitoring, Spectral Indices, Deep Learning Segmentation, and Vector GIS Alerting.

---

## 🛰️ Architecture & Scientific Pipeline Overview

The platform implements an end-to-end scientific pipeline transforming bi-temporal multi-spectral satellite imagery into calibrated geospatial alerts:

```
Earlier Satellite Image (T₁) + Later Satellite Image (T₂)
                        ↓
            ⚙️ Preprocessing (src/preprocess.py)
   (CRS Validation, Co-Registration, Cloud Masking, Normalization)
                        ↓
         🌿 Spectral Indices (src/indices.py)
       (NDVI, EVI, SAVI, NDWI, NBR, dNBR, RdNBR)
                        ↓
     🤖 Deep Neural Inference (src/models/, src/inference.py)
   (ForestUNet, Siamese Change Detector, Early Fusion 4-Class)
                        ↓
         🌲 Forest Segmentation & Deforestation Mask
                        ↓
      🔥 Disturbance Cause Analysis & Severity Assessment
       (Mechanical vs Fire Scars, Empirical Severity Tiers)
                        ↓
       🗺️ Geospatial Vectorization & Export (src/geo.py)
          (Active MMU Filter, GeoJSON, GeoTIFF, Shapefile)
                        ↓
      📈 Multi-Year Persistence Modeling (src/temporal.py)
               (2019 → 2024 Disturbance Regimes)
```

---

## 📁 Repository Structure

```text
deforestation/
├── app.py                     # Streamlit Operational Dashboard (UI Only)
├── train.py                   # PyTorch Model Training & Evaluation Benchmark Script
├── requirements.txt           # Environment Dependencies
├── README.md                  # Project Documentation & Run Instructions
│
├── dataset/                   # Multi-Spectral Sentinel-2 Dataset (5 Bands: B, G, R, NIR, SWIR)
│   ├── train/                 # 16 Bi-Temporal Training Scenes (before/, after/, masks/)
│   ├── validation/            # 4 Validation Scenes
│   └── test/                  # 4 Independent Test Scenes
│
├── models/                    # Verified Trained Model Weights & Checkpoints
│   ├── random_forest_baseline.joblib  # 18-Feature Scikit-Learn Pixel Classifier
│   ├── unet_forest.pth                # PyTorch ForestUNet Semantic Segmentation
│   ├── siamese_change.pth             # PyTorch Dual-Branch Siamese Change Detector
│   └── early_fusion_4class.pth        # PyTorch 10-Channel 4-Class Transition Network
│
├── results/                   # Computed Model Benchmarks & Visual Diagnostics
│   ├── model_evaluation_metrics.json  # Real Test IoU, Dice, Precision, Recall, FPR, FNR
│   ├── ablation_study.json            # Hyperparameter & Loss Function Ablation Table
│   ├── training_curves.png            # PyTorch Loss Convergence Plots
│   └── confusion_matrices.png         # Pixel Confusion Matrix
│
├── src/                       # Core Pipeline Modules
│   ├── preprocess.py          # Rasterio Ingestion, Co-Registration, Cloud Masking
│   ├── indices.py             # Vectorized Spectral Indices (NDVI, NBR, EVI, SAVI, NDWI)
│   ├── inference.py           # Model Inference Engine & Grad-CAM Attention Maps
│   ├── geo.py                 # MMU Filtering, Vectorization, Road Encroachment, GIS Export
│   ├── temporal.py            # Multi-Year Persistence Modeling (2019-2024)
│   └── models/                # Neural Network Definitions
│       ├── baselines.py       # NDVI Differencing & Random Forest
│       ├── unet.py            # ForestUNet with Skip Connections
│       ├── siamese.py         # Siamese Change Detector (Shared Weights)
│       └── early_fusion.py    # Early Fusion 4-Class Network
│
└── outputs/                   # Exported Shapefiles, COGs, and GeoJSONs
```

---

## ⚡ Quick Start & Installation

### 1. Environment Setup
```bash
# Clone repository and navigate to root
cd deforestation

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Evaluation & Benchmarking
Run the training and benchmark script to evaluate all models against the independent test set:
```bash
python train.py
```
This produces `results/model_evaluation_metrics.json`, `results/training_curves.png`, `results/confusion_matrices.png`, and `results/ablation_study.json`.

### 3. Launch Operational Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Dashboard Modules & Verification

1. **Top KPI Cards:**
   - Real computed metrics: Forest Area (ha), Deforested Area (ha), Loss %, Changed Regions (polygons), Loss Severity.
   - Zero hardcoded numbers; all demo badges removed.

2. **Sidebar Controls:**
   - Ingestion: Toggle between bundled test scenes or upload custom Before/After GeoTIFF pairs.
   - Change Threshold ($\tau$): Dynamically filters Siamese change detection sensitivity.
   - Minimum Mapping Unit (ha): Filters noise and polygons below specified acreage.
   - Road Buffer ($m$): Adjusts proximity distance for road encroachment analysis.

3. **Tab 1 — Deforestation Detection Result:**
   - Before/After RGB satellite views with detected crimson canopy loss overlay.
   - Early Fusion 4-Class Transition Map: Forest→Forest, Forest→Cleared, Forest→Burned, Non-Forest→Non-Forest.
   - Cause Analysis: Quantitative breakdown between mechanical clearing vs. wildfire scars via dNBR.
   - Infrastructure Encroachment: Computes % of deforestation within buffer distance of roads.

4. **Tab 2 — Satellite Comparison:**
   - True Color RGB vs. False Color Infrared (CIR) multi-spectral comparisons.

5. **Tab 3 — NDVI & NBR Analysis:**
   - Spectral maps for $T_1$, $T_2$, $\Delta\text{NDVI}$, and $\text{dNBR}$.
   - Bi-temporal histogram distribution shifts.
   - Baseline thresholded change mask responding to sidebar threshold slider.

6. **Tab 4 — Forest Segmentation:**
   - U-Net semantic segmentation mask with an interactive opacity slider.

7. **Tab 5 — Forest Loss Severity:**
   - Empirical quantile-derived disturbance severity map (Tiers 0 to 3).
   - Per-region polygon inspection table with Alert IDs, hectares, and drivers.

8. **Tab 6 — Geospatial Deforestation Map:**
   - Interactive Folium web map rendering vector alerts over Esri World Imagery.
   - Downloads for GeoJSON, georeferenced GeoTIFF mask, and GIS bundles.

9. **Tab 7 — Multi-Year Forest Loss Trend:**
   - Temporal sequence (2019 → 2024) classified via persistence modeling: Stable Forest, Gradual Loss, Rapid Loss, Temporary Dip.

10. **Tab 8 — Model Evaluation & Explainability:**
    - Live test metrics loaded from `results/model_evaluation_metrics.json`.
    - Real PyTorch convergence curves and confusion matrix.
    - Grad-CAM attention heatmap visualizing multi-spectral latent activation.
    - Comprehensive ablation table on learning rates, batch sizes, loss functions, and augmentations.
