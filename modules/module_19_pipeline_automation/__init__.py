"""
Module 19: End-to-End Pipeline Automation & CLI Deployment

Provides:
  - DeforestationPipeline: Single-pair end-to-end inference and analytics pipeline
  - BatchPipelineProcessor: Multi-scene batch processing and regional aggregation
"""

from .pipeline_orchestrator import DeforestationPipeline
from .batch_processor import BatchPipelineProcessor

__all__ = [
    "DeforestationPipeline",
    "BatchPipelineProcessor"
]
