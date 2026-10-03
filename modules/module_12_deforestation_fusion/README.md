# Module 12: Deep Learning Fusion Strategies for Deforestation Detection

## 1. Overview & Research Motivation
When processing bi-temporal satellite image pairs $(T_1, T_2)$ for forest loss monitoring, where and how information is merged determines:
1. **Spectral Discrepancy Sensitivity**: Resilience to illumination changes and sensor calibration drift.
2. **Boundary Precision**: Fidelity of logging road corridors and clearcut edge detection.
3. **Parameter & Computational Efficiency**: Throughput during large-scale continental forest monitoring.

**Module 12** systematically evaluates the 4 dominant multi-temporal fusion paradigms in deep learning remote sensing.

---

## 2. The 4 Fusion Paradigms

```
1. EARLY FUSION (EF)
   [T1 (5ch)] ──┐
                ├──> Stack [10ch] ──> [ Single U-Net Encoder ] ──> [ Bottleneck ] ──> [ Decoder ] ──> Change Mask
   [T2 (5ch)] ──┘

2. LATE FUSION (LF)
   [T1 (5ch)] ──> [ Encoder 1 (Unshared) ] ──┐
                                             ├──> Concat Bottlenecks ──> [ Decoder ] ──> Change Mask
   [T2 (5ch)] ──> [ Encoder 2 (Unshared) ] ──┘

3. SIAMESE DIFFERENCING (Siam-Diff)
   [T1 (5ch)] ──> [ Shared Encoder ] ──┬──> |f1 - f2| Diff Skips ──┐
                                       │                           ├──> [ Decoder ] ──> Change Mask
   [T2 (5ch)] ──> [ Shared Encoder ] ──┴──> |b1 - b2| Diff Bottleneck

4. SIAMESE CONCATENATION (Siam-Concat)
   [T1 (5ch)] ──> [ Shared Encoder ] ──┬──> [f1; f2] Concat Skips ─┐
                                       │                           ├──> [ Decoder ] ──> Change Mask
   [T2 (5ch)] ──> [ Shared Encoder ] ──┴──> [b1; b2] Concat Bottleneck
```

### 2.1 Early Fusion (EF / Image Concatenation)
- **Input**: $X = [T_1; T_2] \in \mathbb{R}^{10 \times H \times W}$.
- **Mechanism**: The 10 spectral bands are fused directly at the first convolutional layer.
- **Advantage**: The network can compute cross-temporal band ratios and differences directly from early layers.
- **Disadvantage**: Prone to overfitting on temporal noise or misalignment between scenes.

### 2.2 Late Fusion (LF / Decision-Level Fusion)
- **Input**: Two unshared encoders $E_{\theta_1}$ and $E_{\theta_2}$.
- **Mechanism**: Encoders process each acquisition independently, concatenating only at the deepest feature bottleneck $B = [b_1; b_2]$.
- **Advantage**: Distinct sensor characteristics or seasonal states can be handled by separate branches.
- **Disadvantage**: Lacks high-resolution skip connections from early layers, leading to fuzzy, blurred boundaries.

### 2.3 Siamese Differencing (Siam-Diff)
- **Input**: Shared encoder $E_\theta$.
- **Mechanism**: Absolute differential skip connections: $D^k = |f_1^k - f_2^k|$.
- **Advantage**: Strict weight sharing guarantees feature space alignment. Difference operations enforce invariance to unchanged forested backgrounds.
- **Disadvantage**: Assumes change manifests primarily as magnitude differences in deep feature space.

### 2.4 Siamese Concatenation (Siam-Concat)
- **Input**: Shared encoder $E_\theta$.
- **Mechanism**: Skips are concatenated: $S^k = [f_1^k; f_2^k]$.
- **Advantage**: Allows the decoder to learn arbitrary non-linear multi-temporal transitions.
- **Disadvantage**: Higher decoder parameter count and channel dimensions.

---

## 3. Loss & Benchmarking
All 4 models are trained under identical data, loss, and optimization constraints:
$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE, weighted}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

Metrics evaluated on unseen test scenes:
- Mean IoU (Jaccard Index)
- Dice Score (F1)
- Pixel Accuracy
- Inference Latency per $128 \times 128$ tile (ms)
- Trainable Parameter Footprint

---

## 4. Key Artifacts
- `01_fusion_strategies_comparison.png`: Comprehensive comparative charts.
- `02_spatial_fusion_predictions.png`: Side-by-side segmentation predictions.
- `03_fusion_error_residuals.png`: True Positive / False Positive / False Negative decomposition.
- `fusion_strategies_leaderboard.csv`: Complete numerical performance matrix.
