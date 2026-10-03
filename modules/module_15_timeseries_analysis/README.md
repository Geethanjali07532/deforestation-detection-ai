# Module 15: Spatial-Temporal Time-Series Analysis & Trend Forecasting

## 1. Overview & Motivation
Deforestation is a dynamic temporal trajectory. In most tropical and temperate frontiers, it does not happen overnight:
1. **Pioneer Stage**: Linear roads penetrate intact wilderness.
2. **Selective Degradation**: Moisture deficit, understory thinning, and microclimate drying cause a progressive decline in vegetation vigor ($\Delta\text{NDVI} < 0$).
3. **Abrupt Clearing**: Bulldozing or burning causes sudden, catastrophic canopy loss.
4. **Post-Disturbance Regimes**: Either permanent pasture/crop conversion (flat line) or secondary forest regeneration (positive recovery slope).

**Module 15** models continuous multi-year satellite trajectories, detects structural breakpoints, and forecasts near-future deforestation frontier risk.

---

## 2. Harmonic Seasonal Decomposition
Any vegetation index time-series $Y(t)$ contains three distinct components:
$$Y(t) = \text{Trend}(t) + \text{Seasonality}(t) + \text{Residual}(t)$$
where seasonality is modeled via harmonic Fourier expansion (order $K=2$):
$$\text{Seasonality}(t) = \sum_{k=1}^K \left[ A_k \cos\left(\frac{2\pi k t}{P}\right) + B_k \sin\left(\frac{2\pi k t}{P}\right) \right]$$
with annual period $P = 12$ months.

---

## 3. Structural Breakpoint Detection (LandTrendr / BFAST)
Detects the exact month $t^*$ when canopy loss shifts the landscape state:
- **Drop Magnitude**: $\Delta = \bar{Y}_{\text{pre}} - \bar{Y}_{\text{post}}$
- **Regime Identification**:
  - If post-break slope $\beta_2 > +0.015$: **Secondary Regrowth / Succession**.
  - If post-break slope $\beta_2 \approx 0$: **Permanent Conversion (Pasture/Cattle)**.
  - If post-break slope $\beta_2 < -0.010$: **Ongoing Canopy Decline**.

---

## 4. Deforestation Vulnerability Index (DVI)
Predicts spatial risk across the remaining intact forest:
$$\text{DVI}(x, y) = 0.45 \cdot e^{-d_{\text{deforest}} / 250} + 0.35 \cdot e^{-d_{\text{road}} / 200} + 0.20 \cdot \Delta\text{NDVI}_{\text{norm}}$$
- Categorized into 4 Frontier Risk Tiers:
  - **Tier 0**: Low / Deep Core Forest ($> 500\text{ m}$ from disturbance).
  - **Tier 1**: Moderate Vulnerability ($200 - 500\text{ m}$).
  - **Tier 2**: High Frontier Risk ($50 - 200\text{ m}$).
  - **Tier 3**: Critical Imminent Risk ($< 50\text{ m}$ from active clearings or roads).

---

## 5. Key Output Artifacts
Saved in `outputs/module_15/`:
- `01_timeseries_trajectory_breakpoints.png`: 4-regime trajectory breakdown with detected break markers.
- `02_harmonic_decomposition_and_forecast.png`: Seasonal decomposition and 12-month forward predictive forecast with 95% confidence intervals.
- `03_spatial_deforestation_frontier_risk_map.png`: Continuous DVI heatmap and categorized frontier risk zoning map.
- `timeseries_risk_summary.json`: Detailed numerical forecast parameters and at-risk area audits.
