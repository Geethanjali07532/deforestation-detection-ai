"""
run_module_02.py

Master execution script for Module 2: Understanding Satellite Imagery.
Executes:
  1. Multispectral GeoTIFF verification / generation
  2. Raster metadata extraction and reporting
  3. Per-band statistical analysis (Red, Green, Blue, NIR, SWIR)
  4. Visualizations:
     - Grayscale individual spectral bands
     - True Color RGB vs False Color CIR vs SWIR Agriculture composites
     - Biophysical Spectral Reflectance curves
     - Band value histograms
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows command line / PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure local module directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.module_02_satellite_imagery.dataset_generator import create_sample_multispectral_geotiff
from modules.module_02_satellite_imagery.spectral_inspector import SatelliteImageryInspector

def main():
    print("\n" + "=" * 70)
    print("[*] MODULE 2: UNDERSTANDING SATELLITE IMAGERY - PRACTICAL RUNNER")
    print("=" * 70)

    # 1. Prepare sample multispectral GeoTIFF
    sample_tif = "data/samples/sentinel2_sample_scene.tif"
    if not os.path.exists(sample_tif):
        print(f"\n[1/4] Generating calibrated multispectral GeoTIFF at: {sample_tif}...")
        create_sample_multispectral_geotiff(sample_tif)
    else:
        print(f"\n[1/4] Found multispectral GeoTIFF at: {sample_tif}")

    # 2. Initialize Inspector
    print("\n[2/4] Initializing SatelliteImageryInspector with Rasterio...")
    inspector = SatelliteImageryInspector(sample_tif)

    # Print metadata & statistics
    inspector.print_metadata()
    inspector.print_band_statistics()

    # 3. Generate Visualizations
    out_dir = "outputs/module_02"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[3/4] Generating spectral visualizations into '{out_dir}/'...")

    plot1 = os.path.join(out_dir, "01_individual_spectral_bands.png")
    inspector.plot_individual_bands(plot1)

    plot2 = os.path.join(out_dir, "02_color_composites_comparison.png")
    inspector.plot_composites(plot2)

    plot3 = os.path.join(out_dir, "03_spectral_reflectance_curves.png")
    inspector.plot_spectral_signatures(plot3)

    plot4 = os.path.join(out_dir, "04_band_histograms.png")
    inspector.plot_histograms(plot4)

    # 4. Concluding Summary
    inspector.close()
    print("\n[4/4] Module 2 execution complete!")
    print("=" * 70)
    print("🎯 Key Insights Learned in Module 2:")
    print("  1. Red & Blue Absorption: Forest chlorophyll absorbs Red & Blue photons for photosynthesis.")
    print("  2. NIR Scattering: Spongy mesophyll plant cells scatter NIR, creating the 'Red Edge' jump.")
    print("  3. False Color CIR (NIR-Red-Green): Forest shines bright red, clearings appear cyan/grey.")
    print("  4. SWIR Moisture Sensitivity: Dry deforested soil reflects high SWIR, healthy canopy absorbs SWIR.")
    print(f"📁 All visual outputs saved in: {os.path.abspath(out_dir)}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
