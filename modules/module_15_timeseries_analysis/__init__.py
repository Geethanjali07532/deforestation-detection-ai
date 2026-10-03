"""
Module 15: Spatial-Temporal Time-Series Analysis & Trend Forecasting

Components:
  - VegetationTrajectoryModeler: Harmonic seasonal decomposition and abrupt structural break detection (LandTrendr/BFAST methodology)
  - DeforestationRiskForecaster: Time-series autoregressive forecasting & spatial deforestation frontier risk mapping
"""

from .trajectory_modeler import VegetationTrajectoryModeler
from .risk_forecaster import DeforestationRiskForecaster

__all__ = [
    "VegetationTrajectoryModeler",
    "DeforestationRiskForecaster"
]
