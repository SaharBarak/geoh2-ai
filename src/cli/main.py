"""
CLI Main Module

Command-line interface for H2 seep detection.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def predict_single(args):
    """Predict on a single image."""
    from src.models import ModelFactory

    print(f"Loading model: {args.model or 'default'}")

    if args.config:
        model = ModelFactory.create_from_yaml(args.config)
    else:
        model = ModelFactory.create_default(args.weights)

    if args.weights and not args.config:
        model.load_weights(args.weights)

    print(f"Device: {model.device}")
    print(f"Processing: {args.image}")

    result = model.predict(args.image)

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Class: {result.class_name}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Is H2 Seep (SCD): {'YES' if result.is_scd(args.threshold) else 'NO'}")
    print("\nAll Probabilities:")
    for name, prob in sorted(result.probabilities.items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        print(f"  {name:15} {prob:6.2%} {bar}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


def predict_batch(args):
    """Predict on multiple images."""
    import glob
    from tqdm import tqdm
    from src.models import ModelFactory

    # Find images
    if args.input.is_dir():
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
        images = []
        for pattern in patterns:
            images.extend(glob.glob(str(args.input / pattern)))
    else:
        images = [str(args.input)]

    if not images:
        print("No images found!")
        return

    print(f"Found {len(images)} images")
    print(f"Loading model...")

    if args.config:
        model = ModelFactory.create_from_yaml(args.config)
    else:
        model = ModelFactory.create_default(args.weights)

    print(f"Device: {model.device}")
    print(f"Processing...")

    # Process in batches
    results = []
    for i in tqdm(range(0, len(images), args.batch_size)):
        batch = images[i:i + args.batch_size]
        batch_results = model.predict_batch(batch)
        results.extend(batch_results)

    # Summary
    scd_count = sum(1 for r in results if r.is_scd(args.threshold))

    print("\n" + "=" * 50)
    print("BATCH RESULTS")
    print("=" * 50)
    print(f"Total processed: {len(results)}")
    print(f"Potential H2 seeps (SCDs): {scd_count} ({100*scd_count/len(results):.1f}%)")
    print(f"Non-SCD structures: {len(results) - scd_count}")

    # Class distribution
    from collections import Counter
    class_dist = Counter(r.class_name for r in results)
    print("\nClass Distribution:")
    for name, count in class_dist.most_common():
        pct = 100 * count / len(results)
        print(f"  {name:15} {count:5} ({pct:5.1f}%)")

    # Save results
    if args.output:
        output_data = {
            "summary": {
                "total": len(results),
                "scd_count": scd_count,
                "threshold": args.threshold,
            },
            "predictions": [
                {**r.to_dict(), "image": img}
                for r, img in zip(results, images)
            ]
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # List high-confidence SCDs
    high_conf_scds = [
        (img, r) for img, r in zip(images, results)
        if r.is_scd(args.threshold) and r.confidence > 0.7
    ]

    if high_conf_scds:
        print(f"\nHigh-confidence SCDs (>{args.threshold:.0%}):")
        for img, r in high_conf_scds[:10]:
            print(f"  {Path(img).name}: {r.confidence:.2%}")
        if len(high_conf_scds) > 10:
            print(f"  ... and {len(high_conf_scds) - 10} more")


def train_model(args):
    """Train a model."""
    from src.models import ModelFactory, ModelConfig
    from src.training import Trainer, TrainingConfig

    print("Initializing training...")

    # Load config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            save_dir=args.output,
        )

    # Create model
    model_config = ModelConfig(
        name="h2_detector",
        architecture=args.architecture,
        num_classes=9,
    )
    model = ModelFactory.create(model_config)

    print(f"Model: {args.architecture}")
    print(f"Device: {model.device}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")

    # Train
    trainer = Trainer(model, config)

    from src.training import (
        EarlyStoppingCallback,
        CheckpointCallback,
        ProgressCallback,
    )

    trainer.add_callback(EarlyStoppingCallback(patience=config.patience))
    trainer.add_callback(CheckpointCallback(save_dir=config.save_dir))
    trainer.add_callback(ProgressCallback())

    result = trainer.train(args.data, epochs=config.epochs)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best accuracy: {result.best_accuracy:.2%}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Best weights: {result.best_weights_path}")


def calculate_indices(args):
    """Calculate spectral indices."""
    import rasterio
    import numpy as np
    from src.preprocessing import SpectralIndexCalculator

    print(f"Loading: {args.input}")

    with rasterio.open(args.input) as src:
        # Load bands
        bands = {}
        band_names = ["B2", "B3", "B4", "B8", "B11", "B12"]

        for i, name in enumerate(band_names[:src.count]):
            bands[name] = src.read(i + 1).astype(np.float32)

        profile = src.profile

    print(f"Loaded {len(bands)} bands: {list(bands.keys())}")

    # Calculate indices
    calculator = SpectralIndexCalculator()

    indices_to_compute = args.indices.split(",")
    results = calculator.compute_multiple(indices_to_compute, bands)

    print(f"\nComputed indices:")
    for name, result in results.items():
        print(f"  {name}: mean={np.nanmean(result.value):.4f}, range={result.valid_range}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, result in results.items():
            out_file = output_path / f"{name}.tif"

            profile.update(
                dtype=rasterio.float32,
                count=1,
            )

            with rasterio.open(out_file, 'w', **profile) as dst:
                dst.write(result.value, 1)

            print(f"Saved: {out_file}")


def serve_api(args):
    """Start the API server."""
    import uvicorn

    print(f"Starting API server on {args.host}:{args.port}")

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="H2 Seep Detection - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single image
  h2detect predict image.jpg

  # Batch prediction
  h2detect batch ./images/ --output results.json

  # Train model
  h2detect train --data dataset.yaml --epochs 50

  # Calculate spectral indices
  h2detect indices sentinel2.tif --indices ndvi,bi

  # Start API server
  h2detect serve --port 8000
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Predict single image")
    pred_parser.add_argument("image", type=Path, help="Image file path")
    pred_parser.add_argument("--weights", "-w", help="Model weights path")
    pred_parser.add_argument("--config", "-c", help="Model config YAML")
    pred_parser.add_argument("--threshold", "-t", type=float, default=0.5)
    pred_parser.add_argument("--output", "-o", help="Output JSON file")
    pred_parser.add_argument("--model", "-m", default="yolov8n")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch prediction")
    batch_parser.add_argument("input", type=Path, help="Input directory or file")
    batch_parser.add_argument("--weights", "-w", help="Model weights path")
    batch_parser.add_argument("--config", "-c", help="Model config YAML")
    batch_parser.add_argument("--threshold", "-t", type=float, default=0.5)
    batch_parser.add_argument("--batch-size", "-b", type=int, default=16)
    batch_parser.add_argument("--output", "-o", help="Output JSON file")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data", "-d", required=True, help="Dataset YAML")
    train_parser.add_argument("--config", "-c", help="Training config YAML")
    train_parser.add_argument("--architecture", "-a", default="yolov8n")
    train_parser.add_argument("--epochs", "-e", type=int, default=50)
    train_parser.add_argument("--batch-size", "-b", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=0.001)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--output", "-o", default="runs/train")

    # Indices command
    idx_parser = subparsers.add_parser("indices", help="Calculate spectral indices")
    idx_parser.add_argument("input", type=Path, help="Input GeoTIFF file")
    idx_parser.add_argument("--indices", "-i", default="ndvi,bi")
    idx_parser.add_argument("--output", "-o", help="Output directory")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", "-p", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command == "predict":
        predict_single(args)
    elif args.command == "batch":
        predict_batch(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "indices":
        calculate_indices(args)
    elif args.command == "serve":
        serve_api(args)
    else:
        parser.print_help()


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
