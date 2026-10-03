# Module 10: Advanced Segmentation Architectures for Satellite Forestry

## 1. Overview & Motivation
Standard U-Net architectures rely on simple skip connections that concatenate raw encoder features directly into the decoder. In multispectral satellite imagery, this can introduce background noise, cloud shadows, and spectral confusion between degraded forest and healthy vegetation. 

**Module 10** evaluates advanced architectural paradigms designed to resolve these challenges:
1. **Attention U-Net**: Integrates **Attention Gates (AGs)** on skip connections to selectively highlight salient forest canopies while suppressing background noise.
2. **DeepLabV3-Lite**: Leverages **Atrous Spatial Pyramid Pooling (ASPP)** with dilated convolutions to capture multi-scale contextual features without losing spatial resolution.
3. **Nested U-Net++**: Utilizes **dense, nested skip pathways** to bridge the semantic gap between high-resolution encoder features and abstract decoder representations.

---

## 2. Theoretical Foundations & Mathematical Formulations

### 2.1 Attention U-Net (Oktay et al., 2018)
Attention Gates (AGs) dynamically modulate the skip connections:
$$\hat{x}^l = \alpha^l \odot x^l$$
where the attention coefficient map $\alpha^l \in [0, 1]$ is computed using a coarser gating signal $g$ from the deeper decoder and the spatial feature map $x^l$ from the encoder:
$$\alpha^l = \sigma_2\left(\psi^T \left(\sigma_1(W_x x^l + W_g g + b_g)\right) + b_\psi\right)$$
- $W_x, W_g, \psi$: $1 \times 1$ linear convolutions.
- $\sigma_1$: ReLU activation; $\sigma_2$: Sigmoid activation.
- **Benefit**: Focuses gradient backpropagation on target forest canopy borders without requiring external bounding boxes.

### 2.2 DeepLabV3-Lite (Chen et al., 2017)
Standard convolutions reduce spatial resolution via repeated downsampling. DeepLab avoids this with **Atrous (Dilated) Convolutions**:
$$y[i] = \sum_k x[i + r \cdot k] w[k]$$
where $r$ is the dilation rate. The effective kernel size expands to:
$$k' = k + (k - 1)(r - 1)$$
The **Atrous Spatial Pyramid Pooling (ASPP)** module applies parallel branches:
1. $1 \times 1$ standard convolution ($r=1$).
2. $3 \times 3$ dilated convolution with $r=2$.
3. $3 \times 3$ dilated convolution with $r=4$.
4. $3 \times 3$ dilated convolution with $r=6$.
5. Global Image-Level Average Pooling $\rightarrow 1 \times 1$ convolution $\rightarrow$ Bilinear upsample.
The features are concatenated and projected to capture both localized canopy boundaries and macro-scale forest tracts.

### 2.3 Nested U-Net++ (Zhou et al., 2018)
Instead of direct skip connections, U-Net++ introduces dense intermediate convolutional nodes $x^{i,j}$:
$$x^{i,j} = \mathcal{H}\left(\left[\left[x^{i,k}\right]_{k=0}^{j-1}, \mathcal{U}(x^{i+1,j-1})\right]\right)$$
- $i$: Downsampling layer index along the encoder.
- $j$: Dense block index along the skip pathway.
- $\mathcal{H}(\cdot)$: Convolutional block followed by activation.
- $\mathcal{U}(\cdot)$: Upsampling layer.
- $[\cdot]$: Channel-wise concatenation.
- **Benefit**: Reduces the semantic discrepancy between encoder and decoder features, accelerating convergence and boosting edge localization.

---

## 3. Loss Formulation
All models are optimized under identical conditions using a compound loss:
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$
where:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon}$$

---

## 4. Architectural Summary & Parameters
| Architecture | Key Mechanism | Parameters | Core Strength |
| :--- | :--- | :--- | :--- |
| **Standard U-Net** | Direct Skip Connections | ~483k | Robust baseline, simple implementation |
| **Attention U-Net** | Attention Gates on Skips | ~491k | Suppresses background noise, sharp canopy edges |
| **DeepLabV3-Lite** | Multi-Scale ASPP (rates 1,2,4,6) | ~214k | Parameter-efficient multi-scale context |
| **Nested U-Net++** | Dense Nested Skip Pathways | ~130k | Compact parameter footprint, high fine-detail fidelity |

---

## 5. Artifacts & Outputs
All benchmarking outputs are saved in `outputs/module_10/`:
- `01_model_comparison_iou_dice.png`: Bar chart of Mean IoU & Dice Score, inference latency, and parameter trade-off.
- `02_attention_gate_visualization.png`: Visual inspection of attention maps across skip depths $\alpha_1, \alpha_2, \alpha_3$.
- `03_architecture_predictions_comparison.png`: Side-by-side segmentation predictions on unseen test tiles.
- `model_comparison_leaderboard.csv`: Complete numerical performance matrix.
