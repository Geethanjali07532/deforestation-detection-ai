# Module 17: Interactive Streamlit / Web Geospatial Application

## 1. Overview & System Architecture
To make the AI deforestation models usable by field rangers, non-technical analysts, and environmental stakeholders, **Module 17** provides a full-featured, responsive, interactive Web GIS Dashboard built with **Streamlit**, **Folium**, **Plotly**, and **Rasterio**.

---

## 2. Key Interactive Capabilities

### 2.1 Multi-Temporal Optical Inspector
- Side-by-side comparison of pre-disturbance ($T_1$) and post-disturbance ($T_2$) optical acquisitions.
- Instant toggling between **True-Color RGB (Bands 3, 2, 1)** and **Standard False-Color (Bands 4, 3, 2)** highlighting healthy chlorophyll reflectance vs. cleared soil.

### 2.2 Vegetation & Spectral Index Explorer
- Real-time heatmaps of NDVI, NBR, NDMI, dNBR, dNDVI, and RdNBR.

### 2.3 Deep Learning Inference & Confidence Thresholding
- Interactive detection threshold slider ($\tau \in [0.10, 0.90]$) allowing real-time adjustment of sensitivity vs. specificity.
- Continuous probability confidence heatmaps and clean binary deforestation masks.

### 2.4 Disturbance Severity & Driver Attribution
- 4-tier USGS classified disturbance severity mapping.
- Anthropogenic driver classification (Roads, Fires, Clearcuts, Logging) with interactive Plotly donut charts.

### 2.5 Web GIS Map & Operational Export
- Embedded full-screen **Folium map** with satellite basemap tiles.
- Clickable vector alert polygons with rich attribute popup cards.
- **One-click downloads**:
  - `deforestation_alerts.geojson` (GeoJSON FeatureCollection)
  - `deforestation_alerts.zip` (ESRI Shapefile bundle: `.shp`, `.shx`, `.dbf`, `.prj`)
  - `deforestation_severity_cog.tif` (Cloud-Optimized GeoTIFF)

### 2.6 Time-Series Trajectory & Predictive Risk Forecasting
- Interactive Plotly multi-year trajectory curve with structural break detection.
- 12-month forward predictive forecast with 95% confidence intervals.
- Spatial Deforestation Vulnerability Index (DVI) frontier risk mapping.

---

## 3. Running the Dashboard Locally
Launch the application with a single command from the project root:
```bash
streamlit run app.py
```
Default URL: `http://localhost:8501`
