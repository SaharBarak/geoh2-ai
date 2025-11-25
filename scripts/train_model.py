#!/usr/bin/env python3
"""
Model Training Script

Train YOLOv8 classifier for H2 seep detection.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Train H2 seep detection model"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        required=True,
        help="Path to dataset YAML file"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--architecture", "-a",
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="Model architecture"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("runs/train"),
        help="Output directory for training results"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cuda, cpu, mps)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("H2 SEEP DETECTION - MODEL TRAINING")
    print("=" * 60)

    # Load training config
    from src.training import TrainingConfig, Trainer
    from src.training import (
        EarlyStoppingCallback,
        CheckpointCallback,
        TensorBoardCallback,
        ProgressCallback,
    )
    from src.models import ModelFactory, ModelConfig

    if args.config:
        config = TrainingConfig.from_yaml(str(args.config))
    else:
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            save_dir=str(args.output),
            device=args.device,
            workers=args.workers,
            seed=args.seed,
        )

    # Create model
    model_config = ModelConfig(
        name="h2_detector",
        architecture=args.architecture,
        num_classes=9,
        device=args.device,
    )

    print(f"\nConfiguration:")
    print(f"  Architecture: {args.architecture}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Patience: {config.patience}")
    print(f"  Output: {config.save_dir}")
    print(f"  Seed: {config.seed}")

    # Initialize model
    print("\nInitializing model...")
    model = ModelFactory.create(model_config)
    print(f"  Device: {model.device}")
    print(f"  Parameters: {model._count_parameters():,}")

    # Setup trainer
    trainer = Trainer(model, config)

    # Add callbacks
    trainer.add_callback(EarlyStoppingCallback(
        patience=config.patience,
        monitor="val_accuracy",
    ))
    trainer.add_callback(CheckpointCallback(
        save_dir=config.save_dir,
        save_best_only=True,
    ))
    trainer.add_callback(TensorBoardCallback(
        log_dir=f"{config.save_dir}/tensorboard",
    ))
    trainer.add_callback(ProgressCallback())

    # Train
    print("\nStarting training...")
    print("-" * 60)

    try:
        result = trainer.train(
            data_yaml=str(args.data),
            epochs=config.epochs,
            batch_size=config.batch_size,
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best accuracy: {result.best_accuracy:.2%}")
        print(f"Best epoch: {result.best_epoch}")
        print(f"Epochs trained: {result.epochs_trained}")
        print(f"\nBest weights: {result.best_weights_path}")
        print(f"Last weights: {result.last_weights_path}")
        print(f"Output directory: {result.save_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
