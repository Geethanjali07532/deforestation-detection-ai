# Module 9: Forest Segmentation Using U-Net

## 📌 Overview
In Module 8, patch classification categorized an entire tile into a single category. However, true deforestation boundaries, winding rivers, and logging roads occur at the **sub-tile, pixel level**.

**Module 9 introduces Semantic Segmentation using U-Net** (Ronneberger et al., 2015). Instead of assigning one label to the whole image, the U-Net performs dense pixel-wise classification:
$$\text{Satellite Image } (5 \times H \times W) \longrightarrow \text{U-Net} \longrightarrow \text{Forest Mask } (1 \times H \times W) \quad (\text{Forest} \rightarrow 1, \, \text{Non-Forest} \rightarrow 0)$$

---

## 🔬 Core U-Net Architecture Principles

```
Input (5x128x128)
   │
   ├──> EncBlock 1 (32ch) ───[Skip 1: 32ch]───────────────────┐
   │         ↓ MaxPool (2x2)                                   │
   ├──> EncBlock 2 (64ch) ───[Skip 2: 64ch]─────────┐          │
   │         ↓ MaxPool (2x2)                         │          │
   ├──> EncBlock 3 (128ch) ──[Skip 3: 128ch]──┐     │          │
   │         ↓ MaxPool (2x2)                   │     │          │
   └──> Bottleneck (256ch)                     │     │          │
             ↓ UpConv (2x2)                    │     │          │
        DecBlock 3 (128ch) <───────────────────┘     │          │
             ↓ UpConv (2x2)                          │          │
        DecBlock 2 (64ch)  <─────────────────────────┘          │
             ↓ UpConv (2x2)                                     │
        DecBlock 1 (32ch)  <────────────────────────────────────┘
             ↓ Conv 1x1
        Binary Forest Mask (1x128x128)
```

### 1. Encoder (Contracting Path)
- Successive blocks of `DoubleConv` (Conv2D $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Conv2D $\rightarrow$ BatchNorm $\rightarrow$ ReLU) followed by $2 \times 2$ Max Pooling.
- Progressively increases receptive field size while learning rich spatial-spectral representations.

### 2. Decoder (Expansive Path)
- Transposed Convolutions (`ConvTranspose2d`) upsample low-resolution semantic maps back to original resolution.
- Successive `DoubleConv` units refine pixel-level segmentation boundaries.

### 3. The Power of Skip Connections
- **The Problem**: Deep pooling operations irreversibly lose high-frequency spatial details (e.g. sharp tree canopy boundaries and narrow 1-pixel logging roads).
- **The U-Net Solution**: Skip connections concatenate high-resolution feature maps from encoder stages directly to their corresponding decoder stages (`torch.cat([d_up, e_skip], dim=1)`). This allows the decoder to combine **semantic context** (what the object is) with **precise spatial localization** (where its boundaries lie).

---

## 📉 Loss Functions & Metrics

### 1. Dice Loss (Soft Dice Coefficient)
Standard Cross-Entropy loss struggles when forest or non-forest pixels are imbalanced. Dice Loss directly optimizes spatial overlap:
$$\text{Dice} = \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}, \qquad \mathcal{L}_{\text{Dice}} = 1 - \text{Dice}$$

### 2. Combined BCE + Dice Loss
Combines smooth pixel-level probability calibration with global shape overlap optimization:
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

### 3. IoU (Intersection over Union / Jaccard Index)
Measures the ratio of the overlap area to the union area:
$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$$

---

## 📊 Benchmark Results on Unseen Test Imagery

| Metric | Score on Test Set | Interpretation |
| :--- | :---: | :--- |
| **Mean IoU (Jaccard)** | **93.07%** | Near-perfect spatial overlap between prediction and ground-truth forest mask. |
| **Dice Score (F1)** | **96.41%** | Outstanding pixel-level harmonic overlap. |
| **Recall (Sensitivity)** | **100.00%** | Zero false negatives (captures 100% of the true forest canopy). |
| **Precision** | **93.07%** | Very low over-segmentation along water/road borders. |
| **Inference Latency** | **60.7 ms per tile** | Real-time dense segmentation on CPU. |

---

## 🛠️ Practical Modules & Scripts

| File | Purpose |
| :--- | :--- |
| [`unet_model.py`](file:///c:/Users/LENOVO/deforestation/modules/module_09_unet_segmentation/unet_model.py) | Full PyTorch implementation of `ForestUNet` with skip connections and multi-band input support. |
| [`losses_metrics.py`](file:///c:/Users/LENOVO/deforestation/modules/module_09_unet_segmentation/losses_metrics.py) | `DiceLoss`, `CombinedBCEDiceLoss`, and `compute_iou_dice`. |
| [`segmentation_dataset.py`](file:///c:/Users/LENOVO/deforestation/modules/module_09_unet_segmentation/segmentation_dataset.py) | PyTorch Dataset slicing multispectral rasters and paired forest masks into uniform tiles. |
| [`trainer.py`](file:///c:/Users/LENOVO/deforestation/modules/module_09_unet_segmentation/trainer.py) | Training and validation loops with Dice and IoU tracking. |
| [`run_module_09.py`](file:///c:/Users/LENOVO/deforestation/run_module_09.py) | Master CLI script training the U-Net, evaluating on test imagery, and generating visual figures. |
| [`09_forest_segmentation_unet.ipynb`](file:///c:/Users/LENOVO/deforestation/notebooks/09_forest_segmentation_unet.ipynb) | Interactive notebook for hands-on exploration. |

---

## 📊 Visual Outputs Generated

Check the [`outputs/module_09/`](file:///c:/Users/LENOVO/deforestation/outputs/module_09/) directory:
1. `01_unet_training_curves.png`: Training vs. Validation Loss and Validation IoU & Dice convergence curves.
2. `02_forest_segmentation_prediction.png`: 4-panel visual evaluation showing Input Satellite RGB, Ground-Truth Mask, Continuous U-Net Sigmoid Probabilities, and Final Binary Predicted Mask.
3. `03_unet_architecture_flow.png`: Architecture diagram showing the Encoder, Bottleneck, Decoder, and High-Resolution Skip Connections.
