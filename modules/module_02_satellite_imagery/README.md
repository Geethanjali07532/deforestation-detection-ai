# Module 2: Understanding Satellite Imagery

## 📌 Overview
Satellite imagery is the core foundation of remote sensing and AI-based deforestation detection. Unlike conventional photography, earth observation satellites (such as **Sentinel-2** and **Landsat 8/9**) record reflected electromagnetic radiation across non-visible spectral regions.

This module covers the physical, spatial, and data principles behind satellite imagery, with a practical inspection of **Red, Green, Blue, Near-Infrared (NIR), and Short-Wave Infrared (SWIR)** bands using `rasterio` and `numpy`.

---

## 🔬 Core Theoretical Concepts

### 1. Raster Data Architecture
- **Grid Layout**: Raster data is organized in rows and columns of discrete cells (pixels), where each cell has a numerical value (Digital Number / Surface Reflectance).
- **Channels/Bands**: A multispectral image is a 3D tensor $(C \times H \times W)$, where $C$ is the number of spectral bands, $H$ is the number of rows (latitude/northing), and $W$ is the number of columns (longitude/easting).
- **Affine Transform**: A matrix mapping pixel indices $(row, col)$ to geographic map coordinates $(x, y)$:
  $$\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} col \\ row \\ 1 \end{bmatrix}$$
  where $a$ is the pixel width, $e$ is the negative pixel height, and $(c, f)$ is the upper-left coordinate.

---

### 2. What is a GeoTIFF?
A **GeoTIFF** is an industry-standard Tagged Image File Format (TIFF) augmented with georeferencing metadata:
- **Coordinate Reference System (CRS)**: Specifies the geodetic datum and projection (e.g., `EPSG:32620` for UTM Zone 20N, WGS 84).
- **Spatial Extent / Bounding Box**: The geographic envelope of the scene `(min_x, min_y, max_x, max_y)`.
- **Pixel Resolution**: Ground Sampling Distance (GSD) per pixel (e.g., 10 meters for Sentinel-2 optical bands).
- **NoData Value**: A reserved value (e.g., `-9999` or `0`) representing missing data, cloud masks, or boundary padding.

---

### 3. Spectral Bands & The Electromagnetic Spectrum

| Band Name | Central Wavelength ($\lambda$) | Primary Biophysical Significance |
| :--- | :--- | :--- |
| **Blue** | ~490 nm | Atmospheric scattering, water penetration, absorbed by chlorophyll-a/b |
| **Green** | ~560 nm | Slight peak in vegetation reflectance (why leaves look green to human eyes) |
| **Red** | ~665 nm | Maximum chlorophyll absorption peak (photosynthetic activity) |
| **Near-Infrared (NIR)** | ~842 nm | High reflectance caused by internal spongy mesophyll cell structure |
| **Short-Wave Infrared (SWIR)** | ~1610 nm / ~2190 nm | Governed by leaf moisture, canopy water content, soil moisture, and dry slash |

---

### 4. Why Vegetation Appears Differently Across Spectral Bands

Healthy green vegetation has a very distinct spectral signature:
1. **Strong Absorption in Red (~665 nm) and Blue (~490 nm)**:
   - Chlorophyll pigments in leaves actively absorb Blue and Red photons to fuel photosynthesis.
   - Consequently, healthy forest canopies appear very **dark** in the Red band (surface reflectance typically < 0.05).
2. **The "Red Edge" (~700 nm – 750 nm)**:
   - There is a steep, sharp increase in reflectance between the Red and NIR regions. This sudden slope is called the *Red Edge* and is an unmistakable indicator of green vegetation.
3. **High Plateau in NIR (~842 nm)**:
   - Leaves do not absorb NIR photons because NIR energy is not used in photosynthesis. If plants absorbed NIR, they would overheat and denature their proteins.
   - The spongy mesophyll cell walls act as internal light-scattering prisms, reflecting 40% to 60% of NIR energy back toward space.
   - As a result, dense forest appears intensely **bright / luminous** in NIR.
4. **Water Absorption in SWIR (~1610 nm & ~2190 nm)**:
   - Liquid water inside living plant cells strongly absorbs SWIR radiation.
   - **Deforested / Cleared Land**: When trees are cut down, the canopy is removed, exposing dry bare soil, wood slash, and degraded ground. Dry soil reflects significantly **more SWIR** energy than moist living canopies.

---

### 5. Resolution Dimensions in Satellite Remote Sensing

- **Spatial Resolution**:
  - The ground size represented by each pixel.
  - *Sentinel-2*: 10 m (RGB + NIR), 20 m (Red Edge + SWIR).
  - *Landsat 8/9*: 30 m (multispectral), 15 m (panchromatic).
  - *PlanetScope*: ~3 m (daily constellation).
- **Temporal Resolution**:
  - The revisit frequency of the satellite over the same geographic coordinates.
  - *Sentinel-2A/2B constellation*: 5 days at the equator.
  - *Landsat 8/9*: 8 days combined.
  - Crucial for multi-temporal change detection and tracking deforestation fronts over weeks, months, and years.
- **Radiometric Resolution**:
  - The bit depth used to record brightness (e.g., 12-bit / 16-bit integers or 32-bit floating point surface reflectance).

---

## 🎨 Multispectral Band Composites

Because human vision is limited to Red, Green, and Blue, we combine bands into RGB color displays:

### 1. True Color (RGB: Red, Green, Blue)
- Shows what the landscape looks like to the human eye.
- Forest appears dark green, clearings appear tan/brown, rivers appear deep blue or muddy brown.

### 2. Color Infrared (CIR) False Color (RGB: NIR, Red, Green)
- **Red channel**: NIR band
- **Green channel**: Red band
- **Blue channel**: Green band
- **Appearance**: Dense forest glows **vibrant red/crimson** due to high NIR reflectance. Cleared land, roads, and bare soil appear **cyan/grey**. Water appears **navy/black**.

### 3. SWIR Agriculture & Moisture Composite (RGB: SWIR, NIR, Red)
- **Red channel**: SWIR band
- **Green channel**: NIR band
- **Blue channel**: Red band
- **Appearance**: Dense vegetation appears **lush green**. Deforested clearings and exposed soil appear **bright orange / copper**. Highly effective for identifying new logging roads and active slash burns.

---

## 🛠️ Practical Hands-on Tools & Scripts

| File | Description |
| :--- | :--- |
| [`dataset_generator.py`](file:///c:/Users/LENOVO/deforestation/modules/module_02_satellite_imagery/dataset_generator.py) | Synthesizes a georeferenced 5-band GeoTIFF with realistic biophysical spectral profiles for forest, deforestation clearings, roads, and rivers. |
| [`spectral_inspector.py`](file:///c:/Users/LENOVO/deforestation/modules/module_02_satellite_imagery/spectral_inspector.py) | Core engine that uses `rasterio` and `numpy` to extract raster metadata, calculate band statistics, generate false-color composites, and plot spectral curves. |
| [`run_module_02.py`](file:///c:/Users/LENOVO/deforestation/run_module_02.py) | Master CLI script to execute Module 2 and generate all visual outputs in `outputs/module_02/`. |
| [`02_understanding_satellite_imagery.ipynb`](file:///c:/Users/LENOVO/deforestation/notebooks/02_understanding_satellite_imagery.ipynb) | Interactive Jupyter Notebook for hands-on exploration and experimentation with spectral bands and an NDVI preview. |

---

## 📊 Visual Outputs Generated

Check the [`outputs/module_02/`](file:///c:/Users/LENOVO/deforestation/outputs/module_02/) directory:
1. `01_individual_spectral_bands.png`: Grayscale display of all 5 bands with reflectance scales.
2. `02_color_composites_comparison.png`: Side-by-side comparison of True Color vs. False Color CIR vs. SWIR Composite.
3. `03_spectral_reflectance_curves.png`: Biophysical curves showing the "Red Edge", chlorophyll dip, and SWIR moisture contrast.
4. `04_band_histograms.png`: Reflectance value distribution histograms for each spectral band.
