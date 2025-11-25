"""
Base Detection Model - Abstract Base Class

Defines the interface for all detection models in the system.
Uses Template Method pattern for consistent prediction pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from . import ModelConfig, PredictionResult


class BaseDetectionModel(ABC):
    """
    Abstract base class for H2 seep detection models.

    Implements Template Method pattern: defines the skeleton of the
    prediction algorithm, deferring specific steps to subclasses.
    """

    def __init__(self, config: ModelConfig, weights_path: Optional[str] = None):
        """
        Initialize the detection model.

        Args:
            config: Model configuration
            weights_path: Optional path to pretrained weights
        """
        self.config = config
        self.weights_path = weights_path
        self._model = None
        self._device = None

    @property
    def device(self) -> str:
        """Get the device the model is running on."""
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    def _resolve_device(self) -> str:
        """Resolve the device to use based on config and availability."""
        import torch

        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.config.device

    @abstractmethod
    def build_model(self):
        """Build the model architecture. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def load_weights(self, weights_path: str):
        """Load pretrained weights. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _forward(self, tensor: np.ndarray) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            tensor: Preprocessed input tensor

        Returns:
            Raw model output (logits or probabilities)
        """
        pass

    def preprocess(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Input image (path or array)

        Returns:
            Preprocessed tensor ready for model
        """
        import cv2

        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()

        # Resize to target size
        img = cv2.resize(img, (self.config.input_size, self.config.input_size))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW format
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess(self, output: np.ndarray, image_path: Optional[str] = None) -> PredictionResult:
        """
        Convert model output to PredictionResult.

        Args:
            output: Raw model output (probabilities)
            image_path: Optional source image path

        Returns:
            Structured prediction result
        """
        # Apply softmax if needed (assumes output is logits)
        if output.min() < 0 or output.max() > 1:
            exp_output = np.exp(output - np.max(output))
            probs = exp_output / exp_output.sum()
        else:
            probs = output

        # Flatten if needed
        probs = probs.flatten()

        # Get predicted class
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        class_name = self.config.class_names[class_id]

        # Build probability dictionary
        probabilities = {
            name: float(prob)
            for name, prob in zip(self.config.class_names, probs)
        }

        return PredictionResult(
            class_name=class_name,
            class_id=class_id,
            confidence=confidence,
            probabilities=probabilities,
            image_path=image_path,
        )

    def predict(self, image: Union[str, Path, np.ndarray]) -> PredictionResult:
        """
        Run prediction on a single image (Template Method).

        Args:
            image: Input image (path or array)

        Returns:
            Prediction result
        """
        # Get image path if provided
        image_path = str(image) if isinstance(image, (str, Path)) else None

        # Template Method: preprocess -> forward -> postprocess
        tensor = self.preprocess(image)
        output = self._forward(tensor)
        result = self.postprocess(output, image_path)

        return result

    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        batch_size: int = 16
    ) -> List[PredictionResult]:
        """
        Run prediction on a batch of images.

        Args:
            images: List of input images
            batch_size: Number of images per batch

        Returns:
            List of prediction results
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self._predict_batch_impl(batch)
            results.extend(batch_results)

        return results

    def _predict_batch_impl(
        self,
        images: List[Union[str, Path, np.ndarray]]
    ) -> List[PredictionResult]:
        """
        Implementation of batch prediction.
        Can be overridden by subclasses for optimized batch processing.
        """
        return [self.predict(img) for img in images]

    def get_model_info(self) -> Dict:
        """Get information about the model."""
        return {
            "name": self.config.name,
            "architecture": self.config.architecture,
            "num_classes": self.config.num_classes,
            "class_names": self.config.class_names,
            "input_size": self.config.input_size,
            "device": self.device,
            "weights_path": self.weights_path,
            "parameters": self._count_parameters() if self._model else 0,
        }

    @abstractmethod
    def _count_parameters(self) -> int:
        """Count trainable parameters. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def export_model(self, output_path: str, format: str = "onnx"):
        """Export model to specified format. Must be implemented by subclasses."""
        pass
