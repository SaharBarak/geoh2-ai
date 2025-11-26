"""
YOLOv8 Classifier - Primary Detection Model

Wraps Ultralytics YOLOv8 for classification of sub-circular depressions.
Based on the methodology from Ginzburg et al. (2025).
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from . import ModelConfig, PredictionResult
from .base_model import BaseDetectionModel


class YOLOv8Classifier(BaseDetectionModel):
    """
    YOLOv8-based classifier for H2 seep detection.

    Uses Ultralytics YOLOv8 in classification mode to classify
    sub-circular depressions into 9 categories.
    """

    def __init__(self, config: ModelConfig, weights_path: Optional[str] = None):
        """
        Initialize YOLOv8 classifier.

        Args:
            config: Model configuration
            weights_path: Optional path to pretrained weights (.pt file)
        """
        super().__init__(config, weights_path)
        self.build_model()

        if weights_path:
            self.load_weights(weights_path)

    def build_model(self):
        """Build YOLOv8 classification model."""
        from ultralytics import YOLO

        # Map architecture name to YOLO model
        arch_map = {
            "yolov8n": "yolov8n-cls.pt",
            "yolov8s": "yolov8s-cls.pt",
            "yolov8m": "yolov8m-cls.pt",
            "yolov8l": "yolov8l-cls.pt",
            "yolov8x": "yolov8x-cls.pt",
        }

        model_name = arch_map.get(self.config.architecture, "yolov8n-cls.pt")

        # Load pretrained YOLO model
        self._model = YOLO(model_name)

        # Set device
        self._model.to(self.device)

    def load_weights(self, weights_path: str):
        """
        Load custom trained weights.

        Args:
            weights_path: Path to weights file (.pt)
        """
        from ultralytics import YOLO

        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        self._model = YOLO(weights_path)
        self._model.to(self.device)
        self.weights_path = weights_path

    def _forward(self, tensor: np.ndarray) -> np.ndarray:
        """
        Forward pass through YOLOv8.

        Args:
            tensor: Preprocessed input tensor (N, C, H, W)

        Returns:
            Class probabilities
        """
        import torch

        # Convert to torch tensor
        x = torch.from_numpy(tensor).to(self.device)

        # Run inference
        with torch.no_grad():
            results = self._model.predict(x, verbose=False, conf=self.config.confidence_threshold)

        # Extract probabilities
        if results and len(results) > 0:
            probs = results[0].probs
            if probs is not None:
                return probs.data.cpu().numpy()

        # Return uniform distribution if no results
        return np.ones(self.config.num_classes) / self.config.num_classes

    def predict(self, image: Union[str, Path, np.ndarray]) -> PredictionResult:
        """
        Run prediction on a single image.

        Uses YOLOv8's built-in preprocessing for better compatibility.

        Args:
            image: Input image (path or array)

        Returns:
            Prediction result
        """
        # Get image path if provided
        image_path = str(image) if isinstance(image, (str, Path)) else None

        # Use YOLO's native predict
        results = self._model.predict(
            image,
            verbose=False,
            conf=self.config.confidence_threshold,
            imgsz=self.config.input_size,
        )

        if results and len(results) > 0 and results[0].probs is not None:
            probs = results[0].probs.data.cpu().numpy()

            # Get predicted class
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])

            # Map to our class names (handle potential mismatch)
            if class_id < len(self.config.class_names):
                class_name = self.config.class_names[class_id]
            else:
                class_name = f"class_{class_id}"

            # Build probability dictionary
            probabilities = {}
            for i, prob in enumerate(probs):
                if i < len(self.config.class_names):
                    probabilities[self.config.class_names[i]] = float(prob)
                else:
                    probabilities[f"class_{i}"] = float(prob)

            return PredictionResult(
                class_name=class_name,
                class_id=class_id,
                confidence=confidence,
                probabilities=probabilities,
                image_path=image_path,
            )

        # Fallback for no results
        return PredictionResult(
            class_name="unknown",
            class_id=-1,
            confidence=0.0,
            probabilities={name: 0.0 for name in self.config.class_names},
            image_path=image_path,
        )

    def _predict_batch_impl(
        self, images: List[Union[str, Path, np.ndarray]]
    ) -> List[PredictionResult]:
        """
        Optimized batch prediction using YOLO's native batching.

        Args:
            images: List of images

        Returns:
            List of prediction results
        """
        # Use YOLO's batch prediction
        results = self._model.predict(
            images,
            verbose=False,
            conf=self.config.confidence_threshold,
            imgsz=self.config.input_size,
        )

        predictions = []
        for i, result in enumerate(results):
            image_path = str(images[i]) if isinstance(images[i], (str, Path)) else None

            if result.probs is not None:
                probs = result.probs.data.cpu().numpy()
                class_id = int(np.argmax(probs))
                confidence = float(probs[class_id])

                if class_id < len(self.config.class_names):
                    class_name = self.config.class_names[class_id]
                else:
                    class_name = f"class_{class_id}"

                probabilities = {}
                for j, prob in enumerate(probs):
                    if j < len(self.config.class_names):
                        probabilities[self.config.class_names[j]] = float(prob)
                    else:
                        probabilities[f"class_{j}"] = float(prob)

                predictions.append(
                    PredictionResult(
                        class_name=class_name,
                        class_id=class_id,
                        confidence=confidence,
                        probabilities=probabilities,
                        image_path=image_path,
                    )
                )
            else:
                predictions.append(
                    PredictionResult(
                        class_name="unknown",
                        class_id=-1,
                        confidence=0.0,
                        probabilities={name: 0.0 for name in self.config.class_names},
                        image_path=image_path,
                    )
                )

        return predictions

    def train_model(
        self,
        data_yaml: str,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        patience: int = 10,
        save_dir: str = "runs/train",
        **kwargs,
    ) -> Dict:
        """
        Train the YOLOv8 classifier.

        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Initial learning rate
            patience: Early stopping patience
            save_dir: Directory to save training outputs
            **kwargs: Additional training arguments

        Returns:
            Training results dictionary
        """
        results = self._model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            lr0=learning_rate,
            patience=patience,
            project=save_dir,
            imgsz=self.config.input_size,
            device=self.device,
            **kwargs,
        )

        return {
            "metrics": {
                "accuracy": results.results_dict.get("metrics/accuracy_top1", 0),
                "top5_accuracy": results.results_dict.get("metrics/accuracy_top5", 0),
            },
            "best_weights": str(results.save_dir / "weights" / "best.pt"),
            "last_weights": str(results.save_dir / "weights" / "last.pt"),
            "save_dir": str(results.save_dir),
        }

    def validate_model(self, data_yaml: str) -> Dict:
        """
        Validate the model on a dataset.

        Args:
            data_yaml: Path to dataset YAML file

        Returns:
            Validation metrics
        """
        results = self._model.val(
            data=data_yaml,
            imgsz=self.config.input_size,
            device=self.device,
        )

        return {
            "accuracy_top1": results.results_dict.get("metrics/accuracy_top1", 0),
            "accuracy_top5": results.results_dict.get("metrics/accuracy_top5", 0),
        }

    def _count_parameters(self) -> int:
        """Count trainable parameters in the model."""
        if self._model is None:
            return 0

        try:
            return sum(p.numel() for p in self._model.model.parameters() if p.requires_grad)
        except Exception:
            # Fallback for different model structures
            return 0

    def export_model(self, output_path: str, format: str = "onnx"):
        """
        Export model to specified format.

        Args:
            output_path: Output file path
            format: Export format ("onnx", "torchscript", "tflite", etc.)
        """
        self._model.export(
            format=format,
            imgsz=self.config.input_size,
        )
