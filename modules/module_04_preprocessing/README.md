# Module 4: Satellite Image Preprocessing Pipeline

## 📌 Overview
Raw satellite imagery cannot be fed directly into deep learning architectures. Satellite sensors capture broad regions across different atmospheric states, orbital angles, and solar illumination conditions. 

This module implements the end-to-end preprocessing pipeline specified in the project roadmap:
```
Raw Satellite Image
       ↓
Cloud / Noise Removal
       ↓
Geometric Alignment (Co-Registration)
       ↓
Band Selection & Stacking
       ↓
Normalization
       ↓
Image Tiles (Chips)
```

---

## 🔬 Core Pipeline Components

### 1. Cloud & Shadow Masking
- **The Challenge**: Clouds and cloud shadows contaminate optical satellite imagery, appearing as false deforestation or obscuring ground activity.
- **Physical Principle**: Atmospheric water vapor and ice crystals in clouds cause intense, uniform scattering across visible bands (Blue, Green, Red), while clear vegetation absorbs Blue and Red.
- **Algorithm**: The [`CloudMasker`](file:///c:/Users/LENOVO/deforestation/modules/module_04_preprocessing/preprocessing_pipeline.py#L32) isolates pixels exceeding physical reflectance thresholds in the Blue channel ($\rho_{\text{blue}} > 0.35$) and visible average ($\bar{\rho}_{\text{vis}} > 0.40$), generating a binary cloud mask and applying neighborhood spatial inpainting.

---

### 2. Noise Removal
- **The Challenge**: Atmospheric transmission anomalies and detector electronics introduce random salt-and-pepper speckle noise.
- **Algorithm**: The [`NoiseFilter`](file:///c:/Users/LENOVO/deforestation/modules/module_04_preprocessing/preprocessing_pipeline.py#L67) applies an edge-preserving $3 \times 3$ median filter across each spectral band. Unlike linear Gaussian blurring, median filtering eliminates isolated noisy outliers while strictly preserving sharp, linear logging road edges and deforestation borders.

---

### 3. Geometric Alignment (Co-Registration)
- **The Challenge**: Multi-temporal change detection requires pixel-to-pixel correspondence. If an image captured in 2020 is shifted by even 2 pixels relative to 2023, the model will mistakenly detect the entire scene's edges as deforestation (false positives).
- **Algorithm**: The [`GeometricAligner`](file:///c:/Users/LENOVO/deforestation/modules/module_04_preprocessing/preprocessing_pipeline.py#L82) calculates 2D Fourier Phase Cross-Correlation on the Near-Infrared (NIR) band:
  $$R = \frac{\mathcal{F}\{I_{T_1}\} \cdot \mathcal{F}^*\{I_{T_2}\}}{\left| \mathcal{F}\{I_{T_1}\} \cdot \mathcal{F}^*\{I_{T_2}\} \right|}$$
  The inverse Fourier transform $\mathcal{F}^{-1}\{R\}$ yields an impulse peak whose coordinates $(\Delta y, \Delta x)$ indicate the exact spatial drift. The After image is then geometrically translated to achieve sub-pixel co-registration.

---

### 4. Band Stacking (Temporal Tensors)
- **Early-Fusion Architecture**: Stacks the multi-temporal bands into a single input tensor of shape:
  $$(C_{\text{before}} + C_{\text{after}}, H, W)$$
  For our 5-band imagery (Blue, Green, Red, NIR, SWIR), this yields a **10-channel tensor** allowing convolutional kernels to learn cross-temporal differential features in the very first network layers.

---

### 5. Radiometric Normalization
- Converts raw surface reflectance into bounded numerical distributions suitable for neural network weights:
  - **Robust Percentile Scaling (Default)**: Clips values to the 2nd and 98th percentiles:
    $$x_{\text{norm}} = \text{clip}\left(\frac{x - P_2}{P_{98} - P_2}, 0.0, 1.0\right)$$
  - **Min-Max Scaling**: Linear scaling between minimum and maximum.
  - **Z-Score Normalization**: Standardizes to zero mean and unit variance: $(x - \mu) / \sigma$.

---

### 6. Image Tiling (Chipping)
- Satellite scenes (e.g. $10,000 \times 10,000$ pixels) exceed GPU memory capacity.
- The [`RasterTiler`](file:///c:/Users/LENOVO/deforestation/modules/module_04_preprocessing/preprocessing_pipeline.py#L189) partitions large scenes and ground-truth masks into uniform chips (e.g. $128 \times 128$ or $256 \times 256$) with optional stride/overlap, recording $(row, col)$ offsets for future seamless reconstruction.

---

## 🛠️ Practical Modules & Scripts

| File | Purpose |
| :--- | :--- |
| [`preprocessing_pipeline.py`](file:///c:/Users/LENOVO/deforestation/modules/module_04_preprocessing/preprocessing_pipeline.py) | Full production class implementing `CloudMasker`, `NoiseFilter`, `GeometricAligner`, `RasterNormalizer`, `BandStacker`, and `RasterTiler`. |
| [`run_module_04.py`](file:///c:/Users/LENOVO/deforestation/run_module_04.py) | Master CLI script executing all preprocessing stages on test pairs and generating visual diagnostics. |
| [`04_satellite_image_preprocessing.ipynb`](file:///c:/Users/LENOVO/deforestation/notebooks/04_satellite_image_preprocessing.ipynb) | Interactive notebook for step-by-step experimentation. |

---

## 📊 Outputs Generated

Visual products are saved in [`outputs/module_04/`](file:///c:/Users/LENOVO/deforestation/outputs/module_04/):
1. `01_preprocessing_pipeline_stages.png`: 6-panel transformation showing raw cloud artifact, cloud mask detection, denoising, misaligned image, co-registered alignment, and normalized temporal NIR differences.
2. `02_tiled_chips_inspection.png`: Model-ready $128 \times 128$ chips extracted from the preprocessed scene with synchronized deforestation masks.
