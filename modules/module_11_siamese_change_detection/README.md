# Module 11: Multi-Temporal Change Detection Using Siamese Neural Networks

## 1. Overview & Objective
Traditional change detection methods (such as Image Differencing or $\Delta\text{NDVI}$ thresholding from Module 6) rely on pixel-level mathematical subtractions. While fast, they are susceptible to:
1. **Seasonal phenological drift** (natural autumn leaf shedding mistaken for clearing).
2. **Atmospheric/Illumination discrepancies** (solar elevation angles, thin clouds, mountain shadows).
3. **Agricultural cycle fluctuations** (seasonal harvesting vs. permanent rainforest destruction).

**Module 11** introduces **Siamese Deep Neural Networks**, which learn invariant spatial-temporal representations. A dual-stream architecture with shared weights extracts deep hierarchical features from both pre-disturbance ($T_1$) and post-disturbance ($T_2$) acquisitions, allowing the network to distinguish genuine forest canopy clearing from ambient spectral noise.

---

## 2. Siamese Architecture Details

### 2.1 Shared-Weight Dual-Branch Encoder ($E_\theta$)
Both images $T_1, T_2 \in \mathbb{R}^{5 \times H \times W}$ are passed through an identical encoder parameterized by shared weights $\theta$:
$$f_1 = E_\theta(T_1), \quad f_2 = E_\theta(T_2)$$
Because the encoder weights are shared:
- Feature spaces are strictly aligned and comparable.
- Number of encoder parameters is halved compared to separate two-stream networks.
- Invariant representations of spectral bands are enforced across different acquisition dates.

### 2.2 Multi-Scale Differential Feature Fusion
At each resolution level $k \in \{1, 2, 3, \text{bottleneck}\}$, we calculate absolute differential feature tensors:
$$D^k = \left| f_1^k - f_2^k \right|$$
These difference maps capture:
- **Level 1 ($128 \times 128$)**: Fine boundary disruptions, logging road margins, tree perimeter loss.
- **Level 2 ($64 \times 64$)**: Textural shifts, crown canopy density reductions.
- **Level 3 ($32 \times 32$)**: Broad clearing patches and sub-canopy clearing.
- **Bottleneck ($16 \times 16$)**: High-level semantic deforestation state changes.

### 2.3 Progressive Change Decoder
The decoder progressively upsamples and fuses multi-scale difference features via transposed convolutions and skip connections, outputting a continuous probability map $\hat{Y} \in [0, 1]^{H \times W}$.

---

## 3. Loss Formulation
Deforestation is spatially sparse (often occupying $< 15\%$ of landscape pixels). We optimize using **Compound Change Loss**:
$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE, weighted}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$
where:
$$\mathcal{L}_{\text{BCE, weighted}} = -\frac{1}{N} \sum_{i=1}^N \left[ w \cdot y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
with positive class weight $w = 2.0$.

---

## 4. Key Artifacts & Visualizations
Outputs are saved in `outputs/module_11/`:
- `01_siamese_training_curves.png`: Loss, IoU, and Dice score convergence.
- `02_siamese_vs_traditional_change_detection.png`: 6-panel side-by-side comparison with Traditional $\Delta\text{NDVI}$ and ground truth.
- `03_multiscale_differential_features.png`: Visualization of deep feature difference responses across Levels 1–3 and Bottleneck.
- `siamese_change_detection_metrics.json`: Quantitative benchmark metrics.
