"""
Single Image Predictor
High-level interface for making predictions
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from ..models import ModelConfig, PredictionResult, ModelFactory
from ..models.yolo_classifier import YOLOv8Classifier
from ..utils.logger import setup_logger


class H2SeepPredictor:
    """
    High-level predictor for H2 seep detection.
    Handles configuration loading and prediction workflow.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cuda"
    ):
        """
        Initialize predictor.

        Args:
            config_path: Path to model configuration YAML
            weights_path: Path to model weights
            device: Device to run on (cuda/cpu)
        """
        self.logger = setup_logger("H2SeepPredictor")

        # Load configuration
        if config_path:
            config_dict = self.load_config(config_path)
            self.model_config = self._config_dict_to_model_config(
                config_dict, device
            )
        else:
            # Default configuration
            self.model_config = self._default_config(device)

        # Initialize model
        self.logger.info(f"Initializing model: {self.model_config.architecture}")
        self.classifier = YOLOv8Classifier(
            self.model_config,
            weights_path=weights_path
        )

        self.logger.info(f"Model loaded on device: {self.classifier.device}")

    def load_config(self, path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file"""
        with open(path) as f:
            return yaml.safe_load(f)

    def _config_dict_to_model_config(
        self, config_dict: Dict, device: str
    ) -> ModelConfig:
        """Convert config dictionary to ModelConfig"""
        model_cfg = config_dict["model"]

        return ModelConfig(
            name=model_cfg["name"],
            architecture=model_cfg["architecture"],
            num_classes=model_cfg["classes"]["num_classes"],
            class_names=model_cfg["classes"]["class_names"],
            input_size=model_cfg["input_size"],
            confidence_threshold=model_cfg["thresholds"]["classification"],
            device=device
        )

    def _default_config(self, device: str) -> ModelConfig:
        """Create default configuration"""
        return ModelConfig(
            name="yolov8_scd_classifier",
            architecture="yolov8n",
            num_classes=9,
            class_names=[
                "SCD", "fairy_circle", "fairy_fort", "farm_circle",
                "flooded_dune", "impact_crater", "karst", "salt_lake",
                "thermokarst"
            ],
            input_size=640,
            confidence_threshold=0.5,
            device=device
        )

    def predict(
        self,
        image_path: Union[str, Path],
        return_probs: bool = False,
        save_result: bool = False,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Predict class for single image.

        Args:
            image_path: Path to input image
            return_probs: Whether to return full probability distribution
            save_result: Whether to save result to JSON
            output_dir: Directory to save results

        Returns:
            Prediction result dictionary
        """
        self.logger.info(f"Predicting: {image_path}")

        # Run prediction
        result = self.classifier.predict(image_path, return_all_probs=return_probs)

        # Convert to dict if needed
        if not isinstance(result, dict):
            result = result.to_dict()

        # Add image path
        result["image_path"] = str(image_path)

        # Log result
        self.logger.info(
            f"Prediction: {result['class_name']} "
            f"(confidence: {result['confidence']:.2%})"
        )

        # Save if requested
        if save_result:
            self._save_result(result, image_path, output_dir)

        return result

    def _save_result(
        self,
        result: Dict,
        image_path: Path,
        output_dir: Optional[Path]
    ) -> None:
        """Save prediction result to JSON"""
        if output_dir is None:
            output_dir = Path("outputs/predictions")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output filename
        image_path = Path(image_path)
        output_file = output_dir / f"{image_path.stem}_prediction.json"

        # Save
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        self.logger.info(f"Result saved to: {output_file}")

    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 16,
        save_results: bool = False,
        output_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        Predict class for multiple images.

        Args:
            image_paths: List of paths to input images
            batch_size: Number of images per batch
            save_results: Whether to save results to JSON
            output_dir: Directory to save results

        Returns:
            List of prediction result dictionaries
        """
        self.logger.info(f"Batch predicting {len(image_paths)} images")

        results = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            batch_results = self.classifier.predict_batch(batch)

            for path, result in zip(batch, batch_results):
                result_dict = result.to_dict()
                result_dict["image_path"] = str(path)
                results.append(result_dict)

        # Save if requested
        if save_results:
            self._save_batch_results(results, output_dir)

        self.logger.info(f"Batch prediction complete: {len(results)} images")
        return results

    def _save_batch_results(
        self,
        results: List[Dict],
        output_dir: Optional[Path]
    ) -> None:
        """Save batch prediction results to JSON"""
        if output_dir is None:
            output_dir = Path("outputs/predictions")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"batch_predictions_{timestamp}.json"

        # Summary
        scd_count = sum(1 for r in results if r.get("is_scd", False))
        summary = {
            "total": len(results),
            "scd_count": scd_count,
            "timestamp": timestamp,
        }

        # Save
        with open(output_file, "w") as f:
            json.dump({"summary": summary, "predictions": results}, f, indent=2)

        self.logger.info(f"Batch results saved to: {output_file}")

    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return self.classifier.get_model_info()

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        weights_path: Optional[str] = None,
    ) -> "H2SeepPredictor":
        """
        Create predictor from configuration file.

        Args:
            config_path: Path to model config YAML
            weights_path: Optional path to weights

        Returns:
            Configured H2SeepPredictor instance
        """
        return cls(config_path=config_path, weights_path=weights_path)

    @classmethod
    def default(cls, weights_path: Optional[str] = None) -> "H2SeepPredictor":
        """
        Create predictor with default configuration.

        Args:
            weights_path: Optional path to weights

        Returns:
            Default H2SeepPredictor instance
        """
        return cls(weights_path=weights_path)


def main():
    """Command-line interface for predictor"""
    import argparse

    parser = argparse.ArgumentParser(
        description="H2 Seep Detection - Single Image Prediction"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save prediction result to JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = H2SeepPredictor(
        config_path=args.config if Path(args.config).exists() else None,
        weights_path=args.weights,
        device=args.device
    )

    # Run prediction
    result = predictor.predict(
        args.image,
        return_probs=True,
        save_result=args.save,
        output_dir=Path(args.output_dir)
    )

    # Print result
    print("\n" + "="*60)
    print("Prediction Result")
    print("="*60)
    print(f"Image: {result['image_path']}")
    print(f"Class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Is H2 Seep: {result['is_scd']}")
    print("\nProbability Distribution:")
    for class_name, prob in sorted(
        result['probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {class_name:20s}: {prob:.2%}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
