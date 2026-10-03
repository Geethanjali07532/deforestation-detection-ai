# Module 13: Deforestation Severity & Post-Disturbance Classification

## 1. Overview & Objective
Detecting that forest canopy has changed is only the first step. For environmental impact assessment, carbon accounting, and post-disturbance recovery planning, foresters and environmental agencies require **disturbance severity grading**:
- Was the disturbance low-intensity selective timber harvesting?
- Did an understory fire burn the shrub layer while sparing mature tree crowns?
- Or did a stand-replacing wildfire or clearcut eliminate 100% of vegetative biomass?

**Module 13** implements the USGS/USFS fire and disturbance severity standard across multi-temporal satellite imagery.

---

## 2. Spectral Disturbance & Burn Indices

### 2.1 Normalized Burn Ratio (NBR)
Healthy vegetation reflects strongly in the Near-Infrared (NIR) due to spongy mesophyll cellular structure, while absorbing Short-Wave Infrared (SWIR). Disturbed, burned, or cleared land reflects heavily in SWIR due to exposed dry mineral soil and charcoal, while NIR plummets:
$$\text{NBR} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}}$$

### 2.2 Differenced NBR (dNBR)
$$\text{dNBR} = \text{NBR}_{\text{pre}} - \text{NBR}_{\text{post}}$$

### 2.3 Relativized Indices
In low-biomass or sparse pre-fire forest, standard dNBR can under-estimate severity. We compute relativized formulations:
- **Relativized dNBR (RdNBR)** (Miller & Thode, 2007):
  $$\text{RdNBR} = \frac{\text{dNBR}}{\sqrt{|\text{NBR}_{\text{pre}}| + 0.001}}$$
- **Relativized Burn Ratio (RBR)** (Parks et al., 2014):
  $$\text{RBR} = \frac{\text{dNBR}}{\text{NBR}_{\text{pre}} + 1.001}$$

---

## 3. USGS Severity Classification Tiers

| Severity Tier | dNBR Range | Typical Ecological Impact |
| :--- | :---: | :--- |
| **0: Undisturbed / Stable** | $< 0.10$ | Intact mature forest canopy, no structural change |
| **1: Low Severity** | $0.10 - 0.27$ | Selective logging, light canopy thinning, surface leaf litter scorch |
| **2: Moderate Severity** | $0.27 - 0.66$ | Heavy canopy thinning, sub-canopy burning, 30–60% tree mortality |
| **3: High Severity** | $\ge 0.66$ | Stand-replacing clearcut, complete canopy loss, deep mineral soil exposure |

---

## 4. Multi-Class Machine Learning Classifier
Using an 18-feature spectral representation combining raw multi-temporal bands and index differentials ($T_1, T_2, \Delta\text{NBR}, \text{RdNBR}, \text{RBR}, \Delta\text{NDVI}, \Delta\text{NDMI}$), an ensemble classifier predicts discrete severity grades for every pixel across entire satellite scenes.

---

## 5. Key Output Artifacts
Saved in `outputs/module_13/`:
- `01_spectral_severity_indices_dnbr_rdnbr.png`: 6-panel continuous spectral index maps.
- `02_spatial_severity_classification_map.png`: Full-scene 4-tier classified severity map and area pie chart.
- `03_severity_confusion_matrix_feature_importance.png`: Multi-class confusion matrix and top spectral predictors.
- `severity_classification_metrics.json`: Area statistics in hectares and classification accuracy report.
