"""
dashboard_components.py

Module 17: Interactive Streamlit / Web Geospatial Application
Provides reusable visual and mapping components:
  - Folium interactive map with satellite tiles and vector alert overlays
  - Plotly interactive figures for driver breakdowns and time-series forecasts
  - KPI metric card formats
"""

import os
import json
from typing import Dict, List, Any, Optional
import folium
import folium.plugins
import plotly.graph_objects as go
import plotly.express as px


DRIVER_COLORS = {
    "Road Encroachment": "#fbc02d",
    "Wildfire Scar": "#d32f2f",
    "Selective Logging": "#00bcd4",
    "Agricultural Clearcut": "#ff6f00",
    "Unassigned": "#9e9e9e"
}


def create_interactive_folium_map(
    geojson_data: Dict[str, Any],
    center_lat: float = -3.12,
    center_lon: float = -60.02,
    zoom_start: int = 13
) -> folium.Map:
    """
    Creates an interactive Folium web map with satellite imagery backdrop
    and vector deforestation alert polygons.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True
    )

    # 1. Base Layer: Esri World Imagery (Satellite)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Imagery (Esri)",
        overlay=False,
        control=True
    ).add_to(m)

    # 2. Alternative Base Layer: OpenStreetMap
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Standard Street Map",
        overlay=False,
        control=True
    ).add_to(m)

    # 3. Style function for GeoJSON polygons
    def style_fn(feature):
        props = feature.get("properties", {})
        driver = props.get("driver", "Unassigned")
        color = DRIVER_COLORS.get(driver, "#ff9100")
        return {
            "fillColor": color,
            "color": "#000000",
            "weight": 1.5,
            "fillOpacity": 0.55
        }

    def highlight_fn(feature):
        return {
            "fillColor": "#ffffff",
            "color": "#ffffff",
            "weight": 3,
            "fillOpacity": 0.8
        }

    # 4. Add GeoJSON Layer with tooltips and popups
    if geojson_data and "features" in geojson_data and len(geojson_data["features"]) > 0:
        folium.GeoJson(
            geojson_data,
            name="Deforestation Alerts (Vector Polygons)",
            style_function=style_fn,
            highlight_function=highlight_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["alert_id", "area_ha", "severity", "driver"],
                aliases=["Alert ID:", "Area (ha):", "Severity:", "Driver:"],
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=["alert_id", "area_ha", "perimeter_m", "severity", "driver", "mean_dnbr", "status"],
                aliases=["Alert ID:", "Area (ha):", "Perimeter (m):", "Severity Tier:", "Disturbance Cause:", "Mean dNBR:", "Status:"]
            )
        ).add_to(m)

    folium.LayerControl(position="topright").add_to(m)
    folium.plugins.Fullscreen(position="topleft").add_to(m)

    return m


def create_driver_pie_chart(driver_stats: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Creates an interactive Plotly donut chart showing deforestation driver breakdown."""
    labels, values, colors = [], [], []

    for driver, s in driver_stats.items():
        area = s.get("total_area_ha", 0.0)
        if area > 0:
            labels.append(driver)
            values.append(area)
            colors.append(DRIVER_COLORS.get(driver, "#888888"))

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#000000", width=1.5)),
        textinfo="label+percent",
        textfont=dict(size=12, family="Arial", color="white")
    )])

    fig.update_layout(
        title="Disturbance Driver Breakdown (Hectares)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig


def create_timeseries_plotly_figure(forecast_dict: Dict[str, Any]) -> go.Figure:
    """Creates an interactive Plotly trajectory and 12-month predictive forecast chart."""
    hist_t = forecast_dict["historical_t"]
    hist_vals = forecast_dict["historical_values"]
    fut_t = forecast_dict["future_t"]
    fut_vals = forecast_dict["forecast"]
    lower_95 = forecast_dict["lower_95"]
    upper_95 = forecast_dict["upper_95"]

    fig = go.Figure()

    # Historical observations
    fig.add_trace(go.Scatter(
        x=hist_t,
        y=hist_vals,
        mode="lines+markers",
        name="Historical Observations",
        line=dict(color="#00e676", width=2.5),
        marker=dict(size=5, color="#00e676")
    ))

    # Upper confidence boundary (transparent fill)
    fig.add_trace(go.Scatter(
        x=fut_t,
        y=upper_95,
        mode="lines",
        name="95% Upper Bound",
        line=dict(width=0),
        showlegend=False
    ))

    # Lower confidence boundary with shaded fill
    fig.add_trace(go.Scatter(
        x=fut_t,
        y=lower_95,
        mode="lines",
        name="95% Confidence Interval",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255, 23, 68, 0.25)"
    ))

    # 12-Month forward forecast
    fig.add_trace(go.Scatter(
        x=fut_t,
        y=fut_vals,
        mode="lines+markers",
        name="12-Month Predictive Forecast",
        line=dict(color="#ff1744", width=3, dash="dash"),
        marker=dict(size=6, color="#ff1744")
    ))

    # Threshold marker
    fig.add_hline(y=0.50, line_dash="dot", line_color="orange", annotation_text="Forest Threshold (NDVI 0.50)")

    fig.update_layout(
        title=f"Canopy Trajectory & 12-Month Forecast (Trend: {forecast_dict.get('projected_annual_rate_pct', 0.0):.2f}%/yr)",
        xaxis_title="Timeline (Months)",
        yaxis_title="Vegetation Index (NDVI)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig


def render_kpi_metrics(
    total_forest_ha: float,
    deforested_ha: float,
    active_alerts_count: int,
    dominant_driver: str
) -> Dict[str, Any]:
    """Returns formatted KPI cards."""
    intact_pct = (1.0 - (deforested_ha / max(1e-5, total_forest_ha + deforested_ha))) * 100.0
    return {
        "total_forest_ha": f"{total_forest_ha:,.1f} ha",
        "deforested_ha": f"{deforested_ha:,.1f} ha",
        "intact_pct": f"{intact_pct:.1f}%",
        "alerts_count": f"{active_alerts_count:,} alerts",
        "dominant_driver": dominant_driver
    }
