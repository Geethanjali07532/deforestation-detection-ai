"""
Module 16: Geospatial Vector Polygon Extraction & GIS Export

GIS Interoperability & Vectorization Engine:
  - RasterPolygonizer: Converts raster change masks into georeferenced Shapely/GeoJSON polygons
  - export_geojson: Writes RFC 7946 GeoJSON FeatureCollections
  - export_shapefile: Writes industry-standard ESRI Shapefile bundles (.shp, .shx, .dbf, .prj)
  - export_cog_geotiff: Writes Cloud-Optimized GeoTIFFs (COG) with pyramids and compression
"""

from .vectorizer import RasterPolygonizer
from .gis_exporter import export_geojson, export_shapefile, export_cog_geotiff

__all__ = [
    "RasterPolygonizer",
    "export_geojson",
    "export_shapefile",
    "export_cog_geotiff"
]
