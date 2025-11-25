"""
Ensemble Model - Combines Multiple Models

Implements Composite pattern to treat ensemble as single model.
Supports various aggregation strategies for improved accuracy.
"""

from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np

from . import ModelConfig, PredictionResult
from .base_model import BaseDetectionModel


class AggregationMethod(Enum):
    """Methods for aggregating ensemble predictions."""
    MEAN = "mean"           # Average probabilities
    MAX = "max"             # Maximum probability per class
    VOTING = "voting"       # Majority voting
    WEIGHTED = "weighted"   # Weighted average


class EnsembleModel(BaseDetectionModel):
    """
    Ensemble of multiple detection models.

    Combines predictions from multiple models using various
    aggregation strategies for improved accuracy and robustness.
    """

    def __init__(
        self,
        models: List[BaseDetectionModel],
        aggregation: AggregationMethod = AggregationMethod.MEAN,
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize ensemble model.

        Args:
            models: List of detection models to ensemble
            aggregation: Method for combining predictions
            weights: Optional weights for each model (for WEIGHTED aggregation)
        """
        if not models:
            raise ValueError("Ensemble requires at least one model")

        # Use first model's config as reference
        config = models[0].config
        super().__init__(config)

        self.models = models
        self.aggregation = aggregation

        # Normalize weights if provided
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of models ({len(models)})"
                )
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / len(models)] * len(models)

    def build_model(self):
        """Build is handled by individual models."""
        pass

    def load_weights(self, weights_path: str):
        """Load weights is handled by individual models."""
        pass

    def _forward(self, tensor: np.ndarray) -> np.ndarray:
        """Forward pass through all models and aggregate."""
        all_probs = []

        for model in self.models:
            probs = model._forward(tensor)
            all_probs.append(probs)

        return self._aggregate(all_probs)

    def _aggregate(self, all_probs: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate predictions from all models.

        Args:
            all_probs: List of probability arrays from each model

        Returns:
            Aggregated probabilities
        """
        probs_array = np.array(all_probs)

        if self.aggregation == AggregationMethod.MEAN:
            return np.mean(probs_array, axis=0)

        elif self.aggregation == AggregationMethod.MAX:
            return np.max(probs_array, axis=0)

        elif self.aggregation == AggregationMethod.VOTING:
            # Get predicted class from each model
            votes = np.argmax(probs_array, axis=-1)
            # Count votes for each class
            num_classes = probs_array.shape[-1]
            vote_counts = np.zeros(num_classes)
            for vote in votes:
                vote_counts[vote] += 1
            # Normalize to probabilities
            return vote_counts / len(self.models)

        elif self.aggregation == AggregationMethod.WEIGHTED:
            # Weighted average
            weighted_probs = np.zeros_like(all_probs[0])
            for prob, weight in zip(all_probs, self.weights):
                weighted_probs += prob * weight
            return weighted_probs

        else:
            return np.mean(probs_array, axis=0)

    def predict(self, image: Union[str, np.ndarray]) -> PredictionResult:
        """
        Run ensemble prediction on a single image.

        Args:
            image: Input image (path or array)

        Returns:
            Aggregated prediction result
        """
        from pathlib import Path

        image_path = str(image) if isinstance(image, (str, Path)) else None

        # Get predictions from all models
        all_probs = []
        for model in self.models:
            result = model.predict(image)
            probs = np.array(list(result.probabilities.values()))
            all_probs.append(probs)

        # Aggregate
        aggregated_probs = self._aggregate(all_probs)

        # Build result
        class_id = int(np.argmax(aggregated_probs))
        confidence = float(aggregated_probs[class_id])
        class_name = self.config.class_names[class_id]

        probabilities = {
            name: float(prob)
            for name, prob in zip(self.config.class_names, aggregated_probs)
        }

        return PredictionResult(
            class_name=class_name,
            class_id=class_id,
            confidence=confidence,
            probabilities=probabilities,
            image_path=image_path,
            metadata={
                "ensemble_size": len(self.models),
                "aggregation": self.aggregation.value,
            }
        )

    def _predict_batch_impl(
        self,
        images: List[Union[str, np.ndarray]]
    ) -> List[PredictionResult]:
        """Batch prediction for ensemble."""
        # Get batch predictions from each model
        all_model_results = []
        for model in self.models:
            results = model.predict_batch(images)
            all_model_results.append(results)

        # Aggregate for each image
        final_results = []
        for i in range(len(images)):
            from pathlib import Path

            image_path = str(images[i]) if isinstance(images[i], (str, Path)) else None

            # Collect probabilities for this image from all models
            all_probs = []
            for model_results in all_model_results:
                result = model_results[i]
                probs = np.array(list(result.probabilities.values()))
                all_probs.append(probs)

            # Aggregate
            aggregated_probs = self._aggregate(all_probs)

            class_id = int(np.argmax(aggregated_probs))
            confidence = float(aggregated_probs[class_id])
            class_name = self.config.class_names[class_id]

            probabilities = {
                name: float(prob)
                for name, prob in zip(self.config.class_names, aggregated_probs)
            }

            final_results.append(PredictionResult(
                class_name=class_name,
                class_id=class_id,
                confidence=confidence,
                probabilities=probabilities,
                image_path=image_path,
                metadata={
                    "ensemble_size": len(self.models),
                    "aggregation": self.aggregation.value,
                }
            ))

        return final_results

    def _count_parameters(self) -> int:
        """Count total parameters across all models."""
        return sum(model._count_parameters() for model in self.models)

    def export_model(self, output_path: str, format: str = "onnx"):
        """
        Export ensemble is not directly supported.
        Export individual models instead.
        """
        raise NotImplementedError(
            "Ensemble export not supported. Export individual models."
        )

    def get_model_info(self) -> Dict:
        """Get information about the ensemble."""
        return {
            "name": f"ensemble_{len(self.models)}",
            "num_models": len(self.models),
            "aggregation": self.aggregation.value,
            "weights": self.weights,
            "models": [model.get_model_info() for model in self.models],
            "total_parameters": self._count_parameters(),
        }
