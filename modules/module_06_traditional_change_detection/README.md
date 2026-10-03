# Module 6: Traditional Forest Change Detection

## 📌 Overview
Before implementing complex convolutional neural networks (CNNs) and semantic segmentation architectures (U-Net, DeepLab), remote sensing workflows must establish a rigorous **quantitative baseline**. 

Traditional change detection leverages algebraic and statistical operations directly on calibrated surface reflectance and spectral vegetation indices. This module evaluates traditional **Image Differencing** and **NDVI Differencing**, optimizes detection thresholds, mitigates false positives, and benchmarks pixel-level performance against ground-truth masks.

---

## 🔬 Core Algorithms

### 1. Image Differencing (Spectral Euclidean Distance)
Compares changes across all spectral bands $(B, G, R, \text{NIR}, \text{SWIR})$ simultaneously:
$$\text{Dist} = \sqrt{\sum_{b=1}^{C} \left( \rho_{b, T_2} - \rho_{b, T_1} \right)^2}$$
- **Advantage**: Captures multi-band radiometric transitions.
- **Limitation**: Sensitive to atmospheric haze and seasonal solar angle shifts.

---

### 2. NDVI Differencing ($\Delta \text{NDVI}$)
Calculates the drop in photosynthetic activity between the earlier date ($T_1$) and later date ($T_2$):
$$\Delta \text{NDVI} = \text{NDVI}_{T_1} - \text{NDVI}_{T_2}$$
- **Physical Meaning**:
  - Healthy forest at $T_1$: $\text{NDVI}_{T_1} \approx 0.85$.
  - Cleared / deforested bare soil at $T_2$: $\text{NDVI}_{T_2} \approx 0.15$.
  - Result: $\Delta \text{NDVI} \approx +0.70$ (massive positive shift!).
  - Unchanged forest or water: $\Delta \text{NDVI} \approx 0.0$.

---

### 3. Threshold-Based Detection
A binary decision boundary $\tau$ converts continuous $\Delta \text{NDVI}$ into a binary change mask $M$:
$$M(x, y) = \begin{cases} 1 & \text{if } \Delta \text{NDVI}(x, y) \ge \tau \quad \text{(Deforested)} \\ 0 & \text{if } \Delta \text{NDVI}(x, y) < \tau \quad \text{(Unchanged)} \end{cases}$$

Using the [`TraditionalChangeDetector.find_optimal_threshold`](file:///c:/Users/LENOVO/deforestation/modules/module_06_traditional_change_detection/change_detector.py#L76) method, we swept $\tau \in [0.05, 0.70]$ to maximize the F1-Score:
$$\tau^* = \arg\max_{\tau} F_1(\tau) \approx 0.517$$

---

### 4. Sources of False Positives & Mitigation
- **Causes of False Positives**:
  1. *Phenological Variation*: Normal dry-season canopy foliage thinning.
  2. *Atmospheric Shadows / Cirrus*: Micro-clouds causing localized reflectance dips.
  3. *Registration Misalignments*: Sub-pixel edge shifts along rivers and roads.
- **Morphological Filtering**:
  - *Binary Opening* ($3 \times 3$ kernel): Removes isolated, single-pixel false positives.
  - *Binary Closing* ($3 \times 3$ kernel): Bridges small gaps within true deforestation polygons.

---

### 5. Quantitative Evaluation Metrics

| Metric | Formula | Value on Test Set | Interpretation |
| :--- | :--- | :---: | :--- |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | **98.98%** | Very low over-detection ($0.12\%$ FPR). |
| **Recall (Sensitivity)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | **88.59%** | Identifies the vast majority of true forest loss. |
| **F1-Score (Dice)** | $\frac{2 \cdot P \cdot R}{P + R}$ | **93.50%** | Harmonic mean of Precision and Recall. |
| **IoU (Jaccard Index)**| $\frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$ | **87.79%** | Area of overlap between prediction and ground truth. |
| **Pixel Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{Total}}$ | **98.54%** | Overall percentage of correctly classified pixels. |

---

## 🛠️ Practical Modules & Scripts

| File | Purpose |
| :--- | :--- |
| [`change_detector.py`](file:///c:/Users/LENOVO/deforestation/modules/module_06_traditional_change_detection/change_detector.py) | Implements `TraditionalChangeDetector` and `ChangeEvaluator`. |
| [`run_module_06.py`](file:///c:/Users/LENOVO/deforestation/run_module_06.py) | Master CLI script evaluating the test set, optimizing thresholds, and printing metrics. |
| [`06_traditional_forest_change_detection.ipynb`](file:///c:/Users/LENOVO/deforestation/notebooks/06_traditional_forest_change_detection.ipynb) | Interactive notebook for hands-on threshold exploration and error mapping. |

---

## 📊 Visual Outputs Generated

Check [`outputs/module_06/`](file:///c:/Users/LENOVO/deforestation/outputs/module_06/):
1. `01_differencing_techniques_comparison.png`: Side-by-side comparison of True Color reference, Spectral Euclidean Distance, and NDVI Differencing.
2. `02_threshold_optimization_curve.png`: Decision threshold curve showing Precision, Recall, F1, and IoU trade-offs across candidate thresholds.
3. `03_traditional_change_detection_benchmark.png`: 4-panel evaluation showing Before, After, Ground-Truth Mask, and the RGB Error Map (Green = TP, Red = FP, Blue = FN).
