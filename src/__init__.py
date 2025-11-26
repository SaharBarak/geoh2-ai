"""
H2 Seep Detection System

A deep learning system for identifying natural hydrogen seeps by
classifying sub-circular depressions from satellite imagery.

Based on: Ginzburg et al. (2025) "Identification of Natural Hydrogen Seeps:
Leveraging AI for Automated Classification of Sub-Circular Depressions"

Modules:
    - models: Detection model architectures (YOLOv8, Ensemble)
    - preprocessing: Spectral indices and image processing
    - postprocessing: Morphometric analysis and geological filtering
    - training: Model training infrastructure
    - data: Data acquisition from Sentinel-2 and Google Maps
    - inference: Prediction pipeline
    - utils: Logging, metrics, and visualization
"""

__version__ = "0.1.0"
__author__ = "H2 Seep Detection Team"
__license__ = "MIT"

# Core exports for convenient access
from .models import (
    EnsembleModel,
    ModelConfig,
    ModelFactory,
    PredictionResult,
    YOLOv8Classifier,
    create_model,
)
from .preprocessing import CoordinateHandler, ImageProcessor, IndexResult, SpectralIndexCalculator

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Models
    "ModelConfig",
    "PredictionResult",
    "ModelFactory",
    "YOLOv8Classifier",
    "EnsembleModel",
    "create_model",
    # Preprocessing
    "SpectralIndexCalculator",
    "ImageProcessor",
    "CoordinateHandler",
    "IndexResult",
]
