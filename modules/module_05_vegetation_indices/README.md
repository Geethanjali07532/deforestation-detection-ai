# Module 5: Vegetation Index Analysis

## 📌 Overview
Spectral vegetation indices transform raw multispectral bands into quantitative indicators of green biomass, photosynthetic activity, canopy moisture, and burn damage. By exploiting differential absorption and scattering across the electromagnetic spectrum, vegetation indices allow AI models to reliably distinguish healthy rainforest from deforested clearings, roads, rivers, and wildfire scars.

This module implements the complete suite of indices specified in the project roadmap:
$$\text{Satellite Image} \longrightarrow \text{NDVI Calculation} \longrightarrow \text{Vegetation Map}$$

---

## 🔬 Mathematical Formulations & Remote Sensing Biophysics

### 1. NDVI (Normalized Difference Vegetation Index)
$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
- **Physics**: Chlorophyll absorption is maximized in the **Red** band (~665 nm), while internal leaf spongy mesophyll scatters **NIR** (~842 nm).
- **Dynamic Range**: $[-1.0, 1.0]$.
  - $\text{NDVI} > 0.6$: Dense tropical rainforest canopy.
  - $0.2 \le \text{NDVI} \le 0.5$: Degraded forest, secondary regrowth, or savanna.
  - $0.0 \le \text{NDVI} < 0.2$: Bare dry soil, clearings, asphalt, and logging roads.
  - $\text{NDVI} < 0.0$: Water bodies (rivers, lakes, ocean).

---

### 2. EVI (Enhanced Vegetation Index)
$$\text{EVI} = G \times \frac{\text{NIR} - \text{Red}}{\text{NIR} + C_1 \cdot \text{Red} - C_2 \cdot \text{Blue} + L}$$
- **Parameters**: $G = 2.5, C_1 = 6.0, C_2 = 7.5, L = 1.0$.
- **Why EVI is Critical**: In dense rainforests (e.g. Amazon or Congo basin), NDVI reaches a saturation plateau when Leaf Area Index (LAI) exceeds ~3. EVI maintains linear sensitivity in dense canopies while using the **Blue** band to decouple atmospheric aerosol scattering from the ground signal.

---

### 3. SAVI (Soil-Adjusted Vegetation Index)
$$\text{SAVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + L_{\text{savi}}} \times (1 + L_{\text{savi}})$$
- **Parameter**: $L_{\text{savi}} = 0.5$.
- **Why SAVI is Critical**: At active deforestation fronts, newly felled tree patches expose bright bare soil. Standard NDVI overestimates or underestimates vegetation due to soil brightness variations. SAVI applies a soil-calibration adjustment factor ($L_{\text{savi}}$) to neutralize soil background noise.

---

### 4. NDWI (Normalized Difference Water / Moisture Index)
- **Gao's Canopy Moisture Index**:
  $$\text{NDWI}_{\text{moisture}} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}}$$
  Reflects liquid water content in the canopy. Living rainforest has high moisture ($\text{NDWI} > 0.4$), whereas felled logs and dry cleared ground reflect high SWIR, dropping NDWI near zero or negative.
- **McFeeters' Open Water Index**:
  $$\text{NDWI}_{\text{water}} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$
  Used to mask rivers and open water bodies from terrestrial vegetation.

---

### 5. NBR (Normalized Burn Ratio)
$$\text{NBR} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}}$$
- **Significance for Deforestation**: Wildfires and agricultural slash-and-burn clearings (Module 14) destroy canopy chlorophyll (lowering NIR) while leaving dry charred soil (increasing SWIR). NBR highlights burn scars and fire-driven forest degradation with extreme contrast.

---

## 🎨 Practical Roadmap Workflow: Satellite Image $\rightarrow$ NDVI $\rightarrow$ Vegetation Map

Using [`VegetationClassifier`](file:///c:/Users/LENOVO/deforestation/modules/module_05_vegetation_indices/vegetation_indices.py#L125), continuous NDVI values are binned into calibrated land-cover categories:

| Class ID | Land Cover Class | NDVI Threshold | Remote Sensing Characteristics |
| :---: | :--- | :--- | :--- |
| **0** | **Water / River** | $\text{NDVI} < 0.0$ | Absorbs NIR almost completely; deep blue/cyan |
| **1** | **Cleared / Bare Soil** | $0.0 \le \text{NDVI} < 0.25$ | High Red reflectance, low NIR; dry exposed soil |
| **2** | **Degraded / Edge Forest** | $0.25 \le \text{NDVI} < 0.55$ | Partial canopy, secondary brush, transition buffer |
| **3** | **Dense Forest Canopy** | $\text{NDVI} \ge 0.55$ | Intense photosynthetic activity, high NIR reflectance |

---

## 🛠️ Practical Modules & Scripts

| File | Purpose |
| :--- | :--- |
| [`vegetation_indices.py`](file:///c:/Users/LENOVO/deforestation/modules/module_05_vegetation_indices/vegetation_indices.py) | Complete production classes implementing `VegetationIndexCalculator` and `VegetationClassifier`. |
| [`run_module_05.py`](file:///c:/Users/LENOVO/deforestation/run_module_05.py) | Master CLI script computing all 5 indices, the vegetation map, and temporal $\Delta \text{NDVI}$ change maps. |
| [`05_vegetation_index_analysis.ipynb`](file:///c:/Users/LENOVO/deforestation/notebooks/05_vegetation_index_analysis.ipynb) | Interactive hands-on notebook with step-by-step visualizations. |

---

## 📊 Visual Outputs Generated

Check the [`outputs/module_05/`](file:///c:/Users/LENOVO/deforestation/outputs/module_05/) directory:
1. `01_all_vegetation_indices_comparison.png`: 6-panel comparison of True Color RGB, NDVI, EVI, SAVI, NDWI Moisture, and NBR.
2. `02_ndvi_vegetation_classification_map.png`: Practical roadmap workflow: Satellite Image $\rightarrow$ Continuous NDVI $\rightarrow$ Classified Discrete Vegetation Map.
3. `03_temporal_vegetation_change_analysis.png`: Multi-temporal $\Delta \text{NDVI}$ analysis showing canopy loss in deep red.
