"""
H2 Seep Detection - Models Module

Provides model architectures for classifying sub-circular depressions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ModelArchitecture(Enum):
    """Supported model architectures."""

    YOLOV8N = "yolov8n"
    YOLOV8S = "yolov8s"
    YOLOV8M = "yolov8m"
    YOLOV8L = "yolov8l"
    YOLOV8X = "yolov8x"


# Default class names from research paper (9 classes)
DEFAULT_CLASS_NAMES = [
    "SCD",  # Sub-Circular Depression (H2-related)
    "fairy_circle",  # Namibian fairy circles
    "fairy_fort",  # Irish ring forts
    "farm_circle",  # Agricultural circles
    "flooded_dune",  # Flooded interdune areas
    "impact_crater",  # Meteorite impact structures
    "karst",  # Karst sinkholes
    "salt_lake",  # Circular salt lakes
    "thermokarst",  # Permafrost thaw lakes
]


@dataclass
class ModelConfig:
    """Configuration for detection models."""

    name: str
    architecture: str = "yolov8n"
    num_classes: int = 9
    class_names: List[str] = field(default_factory=lambda: DEFAULT_CLASS_NAMES.copy())
    input_size: int = 640
    confidence_threshold: float = 0.5
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"

    def __post_init__(self):
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"Number of class names ({len(self.class_names)}) "
                f"must match num_classes ({self.num_classes})"
            )


@dataclass(frozen=True)
class PredictionResult:
    """Immutable prediction result from model inference."""

    class_name: str
    class_id: int
    confidence: float
    probabilities: Dict[str, float]
    image_path: Optional[str] = None
    metadata: Optional[Dict] = None

    def is_scd(self, threshold: float = 0.5) -> bool:
        """Check if prediction indicates H2-related SCD."""
        return self.class_name == "SCD" and self.confidence >= threshold

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "image_path": self.image_path,
            "metadata": self.metadata,
            "is_scd": self.is_scd(),
        }


# Imports for convenient access
from .base_model import BaseDetectionModel
from .yolo_classifier import YOLOv8Classifier
from .ensemble import EnsembleModel
from .model_factory import ModelFactory, create_model

__all__ = [
    "ModelConfig",
    "PredictionResult",
    "ModelArchitecture",
    "DEFAULT_CLASS_NAMES",
    "BaseDetectionModel",
    "YOLOv8Classifier",
    "EnsembleModel",
    "ModelFactory",
    "create_model",
]
