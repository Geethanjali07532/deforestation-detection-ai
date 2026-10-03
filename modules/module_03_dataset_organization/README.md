# Module 3: Dataset Collection & Organization

## 📌 Overview
AI systems for deforestation detection rely on **multi-temporal remote sensing datasets**. Comparing satellite images from different points in time ($T_1$ and $T_2$) allows deep neural networks and machine learning models to detect changes in forest canopy, land clearing, road building, and fire scars.

This module details how forest datasets, multi-temporal imagery, land-cover classes, and ground-truth masks are collected, structured, and split for robust model training.

---

## 📚 Core Topics Covered

### 1. Forest Datasets & Environmental Factors
- Forest loss is driven by both intentional human clearing (logging, agriculture, infrastructure) and climate/weather-driven disturbances (wildfires, drought).
- We ingested the **Forest Fire & Weather Dataset** (`forest_fires.csv`), which records Canadian Forest Fire Weather Index (FWI) components:
  - **FFMC (Fine Fuel Moisture Code)**: Surface fuel litter moisture.
  - **DMC (Duff Moisture Code)**: Moderate-depth decomposing organic layer moisture.
  - **DC (Drought Code)**: Deep organic layer drying (long-term seasonal drought).
  - **ISI (Initial Spread Index)**: Fire velocity potential derived from wind speed and FFMC.
- **Severity Classification**:
  - Class 0: No loss / unburned ($0\text{ ha}$)
  - Class 1: Low loss ($\le 5\text{ ha}$)
  - Class 2: Moderate loss ($5 - 25\text{ ha}$)
  - Class 3: Severe loss ($> 25\text{ ha}$)

---

### 2. Multi-Temporal Satellite Imagery
- Instead of classifying a single image $I$, multi-temporal change detection processes an image pair $(I_{T_1}, I_{T_2})$:
  - $I_{T_1}$ (Before): Reference date showing intact forest canopy.
  - $I_{T_2}$ (After): Target date showing subsequent land-cover modifications.
- **Co-Registration**: For accurate pixel-to-pixel comparison, both images must share the exact same spatial extent, pixel grid alignment, and Coordinate Reference System (CRS).

---

### 3. Land-Cover & Deforestation Datasets
- **Land-Cover Classes**:
  1. Dense Forest Canopy (Primary/Secondary Rainforest)
  2. Deforested / Bare Soil (Clearcutting, Slash, Agricultural expansion)
  3. Water Bodies (Rivers, lakes, reservoirs)
  4. Infrastructure (Logging access tracks, arterial roads, settlements)
- **Deforestation Definition**: Transition from **Forest $\rightarrow$ Non-Forest** between $T_1$ and $T_2$.

---

### 4. Image-Label Relationships & Ground-Truth Masks
- For semantic segmentation and deep change detection, labels are **2D pixel masks** aligned with the satellite images:
  - Mask shape: $(H \times W)$, matching image spatial dimensions.
  - Value `0`: Unchanged (Stable Forest, Stable Water, Stable Soil).
  - Value `1`: Newly Deforested Pixels (Direct forest removal between $T_1$ and $T_2$).
- The manifest [`dataset/metadata.csv`](file:///c:/Users/LENOVO/deforestation/dataset/metadata.csv) links each triplet (`before_path`, `after_path`, `mask_path`) with spatial statistics:
  - Deforested pixel count
  - Deforested area percentage
  - Total deforested area in hectares ($1\text{ pixel at 10m resolution} = 100\text{ m}^2 = 0.01\text{ ha}$).

---

### 5. Training / Validation / Test Organization
```
dataset/
├── metadata.csv               # Dataset manifest with spatial statistics
├── train/                     # 16 Pairs (Used for model gradient updates)
│   ├── before/
│   ├── after/
│   └── masks/
├── validation/                # 4 Pairs (Used for hyperparameter tuning & early stopping)
│   ├── before/
│   ├── after/
│   └── masks/
└── test/                      # 4 Pairs (Held-out benchmark for final evaluation)
    ├── before/
    ├── after/
    └── masks/
```

> [!IMPORTANT]
> **Spatial Data Leakage Prevention**:
> In remote sensing, spatial autocorrelation is strong. Adjacent tiles from the same geographic area must NOT be split between train and test sets. Entire geographic scenes or distinct time periods must be isolated to ensure honest evaluation.

---

## 🛠️ Practical Modules & Scripts

| Script | Purpose |
| :--- | :--- |
| [`forest_dataset_analyzer.py`](file:///c:/Users/LENOVO/deforestation/modules/module_03_dataset_organization/forest_dataset_analyzer.py) | Analyzes tabular forest fire risk, environmental correlations, and severity levels. |
| [`dataset_builder.py`](file:///c:/Users/LENOVO/deforestation/modules/module_03_dataset_organization/dataset_builder.py) | Synthesizes authentic multi-temporal 5-band GeoTIFF pairs with pixel-aligned deforestation masks. |
| [`dataset_validator.py`](file:///c:/Users/LENOVO/deforestation/modules/module_03_dataset_organization/dataset_validator.py) | Audits directory integrity, checking for missing files, orphaned masks, or shape mismatches. |
| [`dataset_visualizer.py`](file:///c:/Users/LENOVO/deforestation/modules/module_03_dataset_organization/dataset_visualizer.py) | Produces visual comparison plots of Before RGB/CIR, After RGB/CIR, and Ground-Truth Overlays. |
| [`run_module_03.py`](file:///c:/Users/LENOVO/deforestation/run_module_03.py) | Master runner executing the entire Module 3 workflow. |
| [`03_dataset_collection_and_organization.ipynb`](file:///c:/Users/LENOVO/deforestation/notebooks/03_dataset_collection_and_organization.ipynb) | Interactive notebook for hands-on inspection. |

---

## 📊 Outputs Generated

Check [`outputs/module_03/`](file:///c:/Users/LENOVO/deforestation/outputs/module_03/):
1. `01_forest_fire_dataset_analysis.png`: Environmental correlations, monthly trends, and severity distributions.
2. `02_multitemporal_pair_inspection.png`: 6-panel triplet inspection showing True Color, False Color CIR, Deforestation Mask, and Change Overlay.
