"""
Model Factory - Creates Detection Models

Implements Factory pattern for model instantiation.
Provides a clean interface for creating different model types.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from . import ModelConfig, DEFAULT_CLASS_NAMES
from .base_model import BaseDetectionModel
from .yolo_classifier import YOLOv8Classifier
from .ensemble import EnsembleModel, AggregationMethod


class ModelFactory:
    """
    Factory for creating detection models.

    Provides static methods for creating various model types
    with proper configuration and initialization.
    """

    # Registry of available model types
    _model_registry = {
        "yolov8": YOLOv8Classifier,
        "yolov8n": YOLOv8Classifier,
        "yolov8s": YOLOv8Classifier,
        "yolov8m": YOLOv8Classifier,
        "yolov8l": YOLOv8Classifier,
        "yolov8x": YOLOv8Classifier,
    }

    @classmethod
    def create(
        cls,
        config: Union[ModelConfig, Dict],
        weights_path: Optional[str] = None,
    ) -> BaseDetectionModel:
        """
        Create a detection model from configuration.

        Args:
            config: ModelConfig or dict with configuration
            weights_path: Optional path to pretrained weights

        Returns:
            Configured detection model
        """
        # Convert dict to ModelConfig if needed
        if isinstance(config, dict):
            config = ModelConfig(**config)

        # Get model class from registry
        arch = config.architecture.lower()
        model_class = cls._model_registry.get(arch)

        if model_class is None:
            available = ", ".join(cls._model_registry.keys())
            raise ValueError(
                f"Unknown architecture: {arch}. "
                f"Available: {available}"
            )

        return model_class(config, weights_path)

    @classmethod
    def create_from_yaml(cls, yaml_path: str) -> BaseDetectionModel:
        """
        Create a model from a YAML configuration file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Configured detection model
        """
        import yaml

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Extract model config and weights path
        model_config = config_dict.get("model", config_dict)
        weights_path = config_dict.get("weights_path")

        return cls.create(model_config, weights_path)

    @classmethod
    def create_yolov8(
        cls,
        name: str = "h2_detector",
        size: str = "n",
        num_classes: int = 9,
        class_names: Optional[List[str]] = None,
        weights_path: Optional[str] = None,
        device: str = "auto",
    ) -> YOLOv8Classifier:
        """
        Convenience method for creating YOLOv8 classifier.

        Args:
            name: Model name
            size: Model size (n, s, m, l, x)
            num_classes: Number of output classes
            class_names: List of class names
            weights_path: Optional pretrained weights
            device: Device to use

        Returns:
            YOLOv8Classifier instance
        """
        config = ModelConfig(
            name=name,
            architecture=f"yolov8{size}",
            num_classes=num_classes,
            class_names=class_names or DEFAULT_CLASS_NAMES[:num_classes],
            device=device,
        )

        return YOLOv8Classifier(config, weights_path)

    @classmethod
    def create_ensemble(
        cls,
        model_configs: List[Union[ModelConfig, Dict]],
        weights_paths: Optional[List[str]] = None,
        aggregation: str = "mean",
        model_weights: Optional[List[float]] = None,
    ) -> EnsembleModel:
        """
        Create an ensemble of multiple models.

        Args:
            model_configs: List of model configurations
            weights_paths: Optional list of weights paths for each model
            aggregation: Aggregation method (mean, max, voting, weighted)
            model_weights: Optional weights for weighted aggregation

        Returns:
            EnsembleModel instance
        """
        if weights_paths is None:
            weights_paths = [None] * len(model_configs)

        if len(weights_paths) != len(model_configs):
            raise ValueError(
                f"Number of weights paths ({len(weights_paths)}) must match "
                f"number of configs ({len(model_configs)})"
            )

        # Create individual models
        models = []
        for config, weights in zip(model_configs, weights_paths):
            model = cls.create(config, weights)
            models.append(model)

        # Map aggregation string to enum
        agg_method = AggregationMethod(aggregation.lower())

        return EnsembleModel(
            models=models,
            aggregation=agg_method,
            weights=model_weights,
        )

    @classmethod
    def create_default(cls, weights_path: Optional[str] = None) -> YOLOv8Classifier:
        """
        Create the default model (YOLOv8n with 9 classes).

        Args:
            weights_path: Optional pretrained weights

        Returns:
            Default YOLOv8 classifier
        """
        return cls.create_yolov8(
            name="h2_detector_default",
            size="n",
            num_classes=9,
            class_names=DEFAULT_CLASS_NAMES,
            weights_path=weights_path,
        )

    @classmethod
    def register_model(cls, name: str, model_class: type):
        """
        Register a custom model class.

        Args:
            name: Model type name
            model_class: Model class (must inherit from BaseDetectionModel)
        """
        if not issubclass(model_class, BaseDetectionModel):
            raise TypeError(
                f"Model class must inherit from BaseDetectionModel"
            )
        cls._model_registry[name.lower()] = model_class

    @classmethod
    def list_available(cls) -> List[str]:
        """List available model types."""
        return list(set(cls._model_registry.keys()))


def create_model(
    config: Union[ModelConfig, Dict, str],
    weights_path: Optional[str] = None,
) -> BaseDetectionModel:
    """
    Convenience function for creating models.

    Args:
        config: ModelConfig, dict, or YAML path
        weights_path: Optional pretrained weights

    Returns:
        Detection model
    """
    if isinstance(config, str):
        # Assume it's a YAML path
        if Path(config).exists() and config.endswith(('.yaml', '.yml')):
            return ModelFactory.create_from_yaml(config)
        else:
            # Assume it's an architecture name
            return ModelFactory.create_yolov8(size=config[-1])

    return ModelFactory.create(config, weights_path)
