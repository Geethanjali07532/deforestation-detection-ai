# Module 18: Model Evaluation, Robustness & Uncertainty Quantification

## 📌 Overview
Operational deployment of deep learning models in earth observation requires strict assurances of **reliability, uncertainty quantification, and environmental robustness**. Models operating on multispectral satellite imagery face seasonal variations, sensor thermal noise, atmospheric hazing, and ambiguous boundary pixels between dense canopy and selective logging.

Module 18 equips the system with:
1. **Bayesian Deep Learning via Monte Carlo Dropout (Gal & Ghahramani, 2016)**: Preserves active spatial dropout ($p=0.25$) at test time across $T$ stochastic passes to quantify epistemic model uncertainty $\sigma(x, y)$.
2. **Ambiguity Mask Filtering**: Flags high-uncertainty pixels ($\sigma \ge 0.15$) to prevent false alarms and route uncertain predictions for human analyst review.
3. **Sensor Noise Stress Testing**: Evaluates resilience under additive Gaussian noise $\mathcal{N}(0, \sigma^2)$ representing electronic sensor degradation.
4. **Atmospheric Hazing / Cloud Simulation**: Stress tests model performance against synthetic cirrus cloud haze and optical attenuation.
5. **Expected Calibration Error (ECE) & Reliability Diagrams**: Assesses whether output probabilities accurately reflect true empirical correctness across $M=10$ confidence bins.

---

## 🧮 Theoretical Background

### 1. Epistemic Uncertainty via Monte Carlo Dropout
Given bi-temporal inputs $X = (I_{t1}, I_{t2})$, we perform $T$ stochastic forward passes by sampling dropout masks $z^{(t)} \sim \text{Bernoulli}(1 - p)$:
$$\hat{y}^{(t)} = f_{\hat{W}^{(t)}}(X)$$

The Bayesian predictive mean is:
$$\mu(x, y) = \frac{1}{T} \sum_{t=1}^T \hat{y}^{(t)}(x, y)$$

The epistemic uncertainty (predictive variance) is:
$$\sigma^2(x, y) = \frac{1}{T} \sum_{t=1}^T (\hat{y}^{(t)}(x, y) - \mu(x, y))^2$$

### 2. Expected Calibration Error (ECE)
Grouping predictions into $M$ equally spaced confidence bins $B_m \in (\frac{m-1}{M}, \frac{m}{M}]$:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
$$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 🔬 Benchmark Results

### 1. Robustness Under Sensor Noise
| Noise Level ($\sigma$) | Mean IoU (%) | Dice Score (%) | Pixel Accuracy (%) | Precision (%) | Recall (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.00 (Clean)** | **82.07%** | **90.15%** | **97.97%** | **84.51%** | 96.60% |
| **0.05** | 80.57% | 89.24% | 97.73% | 82.28% | 97.48% |
| **0.10** | 60.54% | 75.42% | 93.78% | 60.91% | 99.00% |
| **0.15** | 21.27% | 35.08% | 64.44% | 21.28% | 99.69% |
| **0.20** | 10.99% | 19.81% | 22.07% | 10.99% | 99.87% |

*Insight:* The Siamese feature difference architecture maintains solid performance ($>80\%$ IoU) up to $\sigma = 0.05$. Higher noise levels require upstream preprocessing (Module 4 `NoiseFilter` and `RasterNormalizer`).

### 2. Epistemic Uncertainty Quantification
- **Bayesian Passes ($T$):** 10
- **Predictive Mean Probability:** [0.453, 0.596]
- **Epistemic Standard Deviation $\sigma$:** Mean = 0.010, Peak = 0.052
- **Ambiguous Pixels ($\sigma \ge 0.15$):** 0.0% on clean test scenes, confirming high model conviction on distinct canopy clearings.

---

## 📁 Artifacts Produced
- `outputs/module_18/01_mc_dropout_uncertainty_maps.png`: 4-panel figure comparing Before RGB, After RGB, Bayesian Mean Probability, and Spatial Epistemic Uncertainty $\sigma(x, y)$.
- `outputs/module_18/02_robustness_stress_test_curves.png`: Performance degradation curves (IoU, Dice, Accuracy) under increasing noise and cloud hazing.
- `outputs/module_18/03_calibration_reliability_diagram.png`: Reliability diagram with confidence vs. accuracy calibration curve and ECE score.
- `outputs/module_18/robustness_and_uncertainty_report.json`: Quantitative audit log for reproducibility.
- `notebooks/18_model_robustness_and_uncertainty.ipynb`: Interactive Jupyter analysis notebook.
