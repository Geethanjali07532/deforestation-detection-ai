# Module 16: Geospatial Vector Polygon Extraction & GIS Export

## 1. Overview & Operational Mandate
For environmental law enforcement, indigenous reserve protection, and zero-deforestation certification (e.g. EU Deforestation Regulation - EUDR), deep learning outputs must seamlessly integrate into standard GIS workflows.

**Module 16** bridges deep learning raster predictions with operational geospatial vector and raster standards:
1. **Raster-to-Vector Polygonization**: Converts binary change masks into georeferenced polygons with topological boundary consistency.
2. **Douglas-Peucker Simplification**: Smooths pixel stair-stepping artifacts while preserving boundary acreage and road corridor geometry.
3. **Multi-Format GIS Export**:
   - **GeoJSON (RFC 7946)**: Open format for Web GIS, Leaflet, Mapbox, and Google Earth Engine.
   - **ESRI Shapefile Bundle (`.shp`, `.shx`, `.dbf`, `.prj`)**: Industry standard for desktop GIS (QGIS, ArcGIS Pro).
   - **Cloud-Optimized GeoTIFF (COG)**: Multi-band raster with internal $128 \times 128$ tiling, DEFLATE compression, and multi-scale pyramid overviews.

---

## 2. Technical Formulations & Architecture

### 2.1 Raster Polygonization via Affine Transform
Converts raster pixel coordinate matrix $[col, row]^T$ to geographic coordinate space $[X, Y]^T$:
$$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} col \\ row \\ 1 \end{bmatrix}$$
where:
- $a, e$: Pixel resolution in $X$ and $Y$ dimensions.
- $c, f$: Top-left bounding coordinate (False Northing/Easting or Origin Longitude/Latitude).
- $b, d$: Rotation parameters (typically 0 for north-up rasters).

### 2.2 Douglas-Peucker Simplification
Given a polygon boundary $V = \{v_1, v_2, \dots, v_n\}$, the algorithm recursively discards intermediate vertices whose perpendicular distance to the chord is less than threshold $\epsilon$:
$$d_\perp(v_i, \overline{v_1 v_n}) < \epsilon$$
- Prevents file bloat from staircase pixel boundaries.
- Preserves topological validity ($is\_valid = \text{True}$).

---

## 3. Attribute Schema Definition
Every exported polygon carries a standardized attribute schema:
- `ALERT_ID`: Unique deforestation tracking tag (e.g. `DEF_2026_0001`).
- `AREA_HA`: Area in hectares ($ha$).
- `PERIM_M`: Boundary perimeter in meters ($m$).
- `SEVERITY`: Disturbance severity tier (Low, Moderate, High).
- `DRIVER`: Attributed cause (Road Encroachment, Wildfire, Clearcut, Logging).
- `MEAN_DNBR`: Mean burn/canopy loss index.
- `CENTROID_X`, `CENTROID_Y`: Center coordinates for GPS field dispatch.
- `STATUS`: Verification state (`Verified Alert`).

---

## 4. Key Output Artifacts
Saved in `outputs/module_16/`:
- `deforestation_alerts.geojson`: Full GeoJSON FeatureCollection.
- `shapefile/deforestation_alerts.shp`: ESRI Shapefile bundle (`.shp`, `.shx`, `.dbf`, `.prj`).
- `deforestation_severity_cog.tif`: Cloud-Optimized GeoTIFF with internal overviews.
- `01_vector_polygons_overlay.png`: High-resolution cartographic satellite overlay.
- `02_gis_attribute_table_summary.png`: Rendered GIS tabular schema.
- `03_cog_pyramid_structure.png`: COG pyramid overviews visualization.
- `gis_export_manifest.json`: Operational metadata report.
