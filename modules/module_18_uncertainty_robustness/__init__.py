"""
Module 18: Model Evaluation, Robustness & Uncertainty Quantification

Components:
  - MCDropoutChangeDetector: Bayesian Monte Carlo Dropout (MCDO) for epistemic uncertainty estimation
  - RobustnessStressTester: Atmospheric noise, cloud haze, and sensor degradation stress testing
  - CalibrationEvaluator: Expected Calibration Error (ECE) and binned reliability diagrams
"""

from .mc_dropout_evaluator import MCDropoutChangeDetector, estimate_epistemic_uncertainty
from .robustness_stress_tester import RobustnessStressTester, compute_calibration_curve

__all__ = [
    "MCDropoutChangeDetector",
    "estimate_epistemic_uncertainty",
    "RobustnessStressTester",
    "compute_calibration_curve"
]
