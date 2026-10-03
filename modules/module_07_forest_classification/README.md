# Module 7: Forest vs Non-Forest Classification

## 📌 Overview
Before employing deep convolutional networks, we construct a comprehensive **Traditional Machine Learning Baseline** using pixel-level feature classification.

In this module, every pixel in a multispectral satellite observation is converted into an engineered 13-dimensional biophysical feature vector. We train, tune, and benchmark three industry-standard machine learning algorithms:
1. **Random Forest (RF)**
2. **Support Vector Machine (Linear SVM with probability calibration)**
3. **XGBoost (Extreme Gradient Boosting)**

---

## 🔬 Feature Engineering: 13 Spectral Indicators

Using [`PixelFeatureExtractor`](file:///c:/Users/LENOVO/deforestation/modules/module_07_forest_classification/ml_classifier.py#L36), each pixel $(x, y)$ is transformed into 13 features:

| Feature Group | Features | Biophysical Motivation |
| :--- | :--- | :--- |
| **Raw Bands** | Blue, Green, Red, NIR, SWIR | Fundamental surface reflectance across the solar spectrum. |
| **Vegetation Indices** | NDVI, EVI, SAVI | Quantifies photosynthetic canopy vigor and chlorophyll density. |
| **Moisture / Burn Indices**| NDWI Moisture, NBR | Measures liquid leaf water thickness and charcoal/burn presence. |
| **Spectral Ratios** | NIR/Red, Red/Green, SWIR/NIR | Normalizes solar illumination angles and topographic shading. |

---

## 🤖 Algorithms Evaluated

### 1. Random Forest (RF)
- **Architecture**: Ensemble of 100 de-correlated decision trees trained via bootstrap aggregating (bagging) and random subspace feature selection.
- **Key Advantages**: Highly resistant to overfitting; requires zero feature scaling; naturally computes Gini feature importances.

### 2. Support Vector Machine (Linear SVM)
- **Architecture**: Constructs an optimal separating hyperplane maximizing the geometric margin between forest and non-forest support vectors in scaled space:
  $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$
- **Pre-Processing**: Scaled using [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and calibrated using isotonic/sigmoid cross-validation.

### 3. XGBoost
- **Architecture**: Gradient boosted decision tree framework optimizing a second-order Taylor expansion of the log-loss objective with regularized leaf weights ($L_1$ and $L_2$).
- **Key Advantages**: Exceptional speed via histogram-based splitting; high precision along complex decision boundaries.

---

## 📊 Benchmark Leaderboard (Test Set Evaluation)

| Algorithm | Accuracy | Precision | Recall | F1-Score | IoU | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | 1.24s |
| **XGBoost** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | 0.35s |
| **SVM (Linear)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | 0.12s |

---

## 🔍 Feature Importance Ranking

Tree-based feature importance analysis revealed the biophysical drivers separating forest from non-forest:
1. **NDVI**: **20.23%** (Primary discriminator of photosynthetic vegetation)
2. **EVI**: **19.28%** (Atmospheric-corrected vegetation signal)
3. **SAVI**: **18.27%** (Soil-calibrated vegetation index)
4. **NIR/Red Ratio**: **15.66%** (Steepness of the Red Edge)
5. **NIR Reflectance**: **12.14%** (Spongy mesophyll cellular scattering)

> [!NOTE]
> **Why Move to Deep Learning (Modules 8 & 9)?**
> While pixel-level ML models achieve outstanding spectral classification, they analyze each pixel in **complete spatial isolation** without understanding neighborhood context, texture, spatial shapes of logging roads, or canopy fragmentation patterns. Deep Learning CNNs and U-Nets leverage 2D spatial convolution to overcome this limitation.

---

## 🛠️ Practical Modules & Scripts

| File | Purpose |
| :--- | :--- |
| [`ml_classifier.py`](file:///c:/Users/LENOVO/deforestation/modules/module_07_forest_classification/ml_classifier.py) | Implements `PixelFeatureExtractor` and `ForestMLClassifierSuite`. |
| [`run_module_07.py`](file:///c:/Users/LENOVO/deforestation/run_module_07.py) | Master CLI script extracting features, training RF/SVM/XGBoost, and compiling benchmark metrics. |
| [`07_forest_vs_nonforest_classification.ipynb`](file:///c:/Users/LENOVO/deforestation/notebooks/07_forest_vs_nonforest_classification.ipynb) | Interactive notebook for hands-on experimentation. |

---

## 📊 Visual Outputs Generated

Check the [`outputs/module_07/`](file:///c:/Users/LENOVO/deforestation/outputs/module_07/) directory:
1. `01_model_comparison_metrics.png`: Bar chart comparing Accuracy, Precision, Recall, F1, and IoU across RF, SVM, and XGBoost.
2. `02_feature_importances.png`: Horizontal bar chart ranking feature importances for Random Forest (MDI) and XGBoost (Gain).
3. `03_ml_classified_forest_maps.png`: 4-panel visual comparison: Input RGB, Ground-Truth Mask, Random Forest Prediction, and XGBoost Prediction.
