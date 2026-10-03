"""
Module 17: Interactive Streamlit / Web Geospatial Application

Interactive Environmental Intelligence Dashboard:
  - Multi-temporal satellite imagery comparison
  - Real-time vegetation & burn index computation
  - Deep Learning Siamese change detection inference
  - Disturbance severity & driver attribution
  - Interactive Folium web GIS map with vector alert overlays
  - Time-series trajectory break detection & 12-month risk forecasting
  - One-click GeoJSON, Shapefile, and COG GeoTIFF export
"""

from .dashboard_components import (
    create_interactive_folium_map,
    render_kpi_metrics,
    create_driver_pie_chart,
    create_timeseries_plotly_figure
)

__all__ = [
    "create_interactive_folium_map",
    "render_kpi_metrics",
    "create_driver_pie_chart",
    "create_timeseries_plotly_figure"
]
