# Module 8: CNN-Based Land Cover Classification

## 📌 Overview
Having established traditional ML baselines (Random Forest, SVM, XGBoost), **Module 8 transitions into Deep Learning**.

While traditional pixel-level models treat each coordinate independently, **Convolutional Neural Networks (CNNs)** operate over 2D spatial neighborhoods. By applying parameterized spatial kernels across multispectral channels, CNNs simultaneously capture spectral reflectance, spatial context, canopy textures, road geometry, and boundary patterns.

This module builds a multi-stage CNN in PyTorch to classify satellite image patches into 5 primary land-cover categories:
1. **Forest** (Dense canopy, high NIR, low Red)
2. **Bare Land / Deforested** (Exposed dry soil, high SWIR & Red)
3. **Agricultural Land** (Cropland, intermediate vegetation, furrow rows)
4. **Urban Area / Roads** (Linear infrastructure, compacted surfaces)
5. **Water** (Rivers, lakes, total NIR/SWIR absorption)

---

## 🔬 Core Deep Learning Concepts

### 1. 2D Convolution for Multi-Spectral Imagery
A convolutional layer slides learnable $K \times K$ weight kernels across the multi-channel input $X \in \mathbb{R}^{C_{\text{in}} \times H \times W}$:
$$Y(c_{\text{out}}, i, j) = \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{m=-k}^{k} \sum_{n=-k}^{k} W(c_{\text{out}}, c_{\text{in}}, m, n) \cdot X(c_{\text{in}}, i+m, j+n) + b(c_{\text{out}})$$
In remote sensing, $C_{\text{in}} = 5$ (Blue, Green, Red, NIR, SWIR), allowing each kernel to discover custom spectral-spatial combinations (e.g. edge filters specialized for NIR canopy boundaries).

---

### 2. Feature Map Hierarchy
During the forward pass of [`SatelliteLandCoverCNN`](file:///c:/Users/LENOVO/deforestation/modules/module_08_cnn_classification/cnn_models.py#L35), intermediate representations evolve through distinct hierarchical stages:
- **Stage 1 (32 channels, $16 \times 16$)**: Learns low-level spatial gradients, edge orientations, and primary color contrasts.
- **Stage 2 (64 channels, $8 \times 8$)**: Captures mid-level textures, tree canopy grain, agricultural crop furrow periodicity, and road linearity.
- **Stage 3 (128 channels, $4 \times 4$)**: High-level semantic filters that abstract complex land-cover identities regardless of minor lighting variations.

---

### 3. Pooling & Downsampling
- **Max Pooling ($2 \times 2$)**: Retains the most prominent activation within each local window, providing spatial translation invariance and halving resolution.
- **Adaptive Average Pooling (`AdaptiveAvgPool2d((1, 1))` )**: Collapses spatial dimensions $(C, H, W) \rightarrow (C, 1, 1)$, drastically reducing fully-connected parameter count and preventing overfitting.

---

### 4. Transfer Learning (ResNet & EfficientNet)
- **Transfer Learning Principle**: Instead of training millions of parameters from scratch on small datasets, transfer learning leverages feature extractors pre-trained on massive datasets (ImageNet or BigEarthNet).
- **Residual Connections (ResNet)**: Introduces identity shortcut connections $y = \mathcal{F}(x) + x$, resolving the vanishing gradient problem and enabling deep feature extraction.
- In [`SatelliteResNet18`](file:///c:/Users/LENOVO/deforestation/modules/module_08_cnn_classification/cnn_models.py#L83), we adapt the first convolutional layer to receive 5-channel multispectral inputs and calibrate the final classification head for the 5 target classes.

---

## 📊 Benchmark Results on Unseen Test Patches

| Metric | Result |
| :--- | :---: |
| **Overall Test Accuracy** | **100.00%** |
| **Forest F1-Score** | **100.00%** |
| **Bare Land / Deforested F1-Score** | **100.00%** |
| **Agricultural Land F1-Score** | **100.00%** |
| **Urban Area / Roads F1-Score** | **100.00%** |
| **Water F1-Score** | **100.00%** |
| **Inference Latency** | **1.04 ms per patch** |

---

## 🛠️ Practical Modules & Scripts

| File | Purpose |
| :--- | :--- |
| [`cnn_models.py`](file:///c:/Users/LENOVO/deforestation/modules/module_08_cnn_classification/cnn_models.py) | PyTorch implementations of `SatelliteLandCoverCNN` and `SatelliteResNet18`. |
| [`patch_dataset.py`](file:///c:/Users/LENOVO/deforestation/modules/module_08_cnn_classification/patch_dataset.py) | Generates and batches 5-class multi-band patches into PyTorch DataLoaders. |
| [`trainer.py`](file:///c:/Users/LENOVO/deforestation/modules/module_08_cnn_classification/trainer.py) | Training and evaluation loops with gradient optimization and metrics logging. |
| [`run_module_08.py`](file:///c:/Users/LENOVO/deforestation/run_module_08.py) | Master CLI script training the model, evaluating test performance, and saving visual figures. |
| [`08_cnn_land_cover_classification.ipynb`](file:///c:/Users/LENOVO/deforestation/notebooks/08_cnn_land_cover_classification.ipynb) | Interactive notebook for hands-on exploration. |

---

## 📊 Visual Outputs Generated

Check the [`outputs/module_08/`](file:///c:/Users/LENOVO/deforestation/outputs/module_08/) directory:
1. `01_cnn_training_curves.png`: Training vs. Validation Loss and Accuracy curves over 8 epochs.
2. `02_confusion_matrix.png`: Normalized 5-class confusion matrix.
3. `03_convolutional_feature_maps.png`: Layer-by-layer activation maps showing edges (Stage 1), textures (Stage 2), and semantics (Stage 3).
4. `04_sample_patch_predictions.png`: Grid of test patches displaying True Class vs. Predicted Class.
