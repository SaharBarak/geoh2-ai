"""
Single Image Predictor
High-level interface for making predictions
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from ..models import ModelConfig
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

    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return self.classifier.get_model_info()


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
