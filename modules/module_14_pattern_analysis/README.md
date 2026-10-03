# Module 14: Fire, Logging & Road Encroachment Pattern Analysis

## 1. Overview & Landscape Ecology Motivation
Deforestation exhibits distinct geometric, spatial, and topological patterns depending on its underlying anthropological or natural cause. 

Understanding the spatial morphology allows environmental agencies to distinguish:
1. **Illegal logging roads and fishbone arteries** penetrating deep into indigenous reserves.
2. **Wildfires and escaped agricultural burns** with irregular, fractal perimeters.
3. **Selective logging punctures** and smallholder clearing gaps.
4. **Industrial-scale agro-industrial clearcuts** (rectilinear geometric boundaries).

Furthermore, deforestation triggers **forest fragmentation** and severe **edge effects** (e.g. microclimate drying, tree mortality, windthrow, invasive species) that extend up to 100 meters into adjacent "intact" canopy.

---

## 2. Morphological Landscape Metrics

### 2.1 Patch Linearity & Aspect Ratio ($L/W$)
Using the oriented minimum bounding rectangle enclosing each connected disturbance polygon:
$$\text{Linearity} = \frac{\max(\text{Length}, \text{Width})}{\min(\text{Length}, \text{Width})}$$
- Roads, transport corridors, and seismic lines exhibit $\text{Linearity} \ge 3.0$.

### 2.2 Circularity / Isoperimetric Quotient ($C$)
Measures compactness relative to a perfect circle:
$$C = \frac{4 \pi \cdot A}{P^2}$$
- Circles yield $C \approx 1.0$.
- Linear transport corridors yield $C \ll 0.15$.

### 2.3 Solidity / Convexity
Measures whether a patch has concave indentations:
$$\text{Solidity} = \frac{A}{A_{\text{convex hull}}}$$
- Geometric industrial clearcuts exhibit $\text{Solidity} \ge 0.75$.
- Uncontrolled fire scars exhibit $\text{Solidity} < 0.60$.

### 2.4 Fractal Dimension ($D$)
Quantifies boundary perimeter complexity (Mandelbrot & Lovejoy):
$$D = \frac{2 \ln(P / 4)}{\ln(A)}$$
- Planar geometric clearings: $D \approx 1.0$.
- Natural fire perimeters influenced by terrain and turbulent wind: $D \ge 1.30$.

---

## 3. Forest Fragmentation & Edge Effect Model
Using Euclidean distance transformation from non-forest edges:
- **Core Interior Forest**: Distance to nearest clearing $> 100\text{ m}$.
- **Degraded Edge Forest**: Distance to nearest clearing $\le 100\text{ m}$.
- **Edge-to-Core Ratio**:
  $$R_{\text{edge/core}} = \frac{A_{\text{edge}}}{A_{\text{core}}}$$
  A rising ratio indicates severe landscape degradation even if total canopy loss is small.

---

## 4. Key Output Artifacts
Saved in `outputs/module_14/`:
- `01_spatial_pattern_driver_map.png`: Multi-panel map overlaying color-coded disturbance driver polygons on satellite imagery.
- `02_morphological_feature_distributions.png`: Scatter plots demonstrating cluster separation among drivers.
- `03_forest_fragmentation_edge_effect.png`: 100m edge buffer map and intactness audit.
- `disturbance_patches_manifest.csv`: Full database of all extracted patches and geometric attributes.
- `pattern_analysis_summary.json`: Landscape statistics and driver breakdowns.
