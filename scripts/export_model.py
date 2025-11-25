#!/usr/bin/env python3
"""
Model Export Script

Export trained models to various formats (ONNX, TorchScript, etc.)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Export H2 seep detection model"
    )
    parser.add_argument(
        "--weights", "-w",
        type=Path,
        required=True,
        help="Path to model weights (.pt file)"
    )
    parser.add_argument(
        "--format", "-f",
        default="onnx",
        choices=["onnx", "torchscript", "tflite", "coreml", "engine"],
        help="Export format"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export with FP16 precision"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic batch size"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("H2 SEEP DETECTION - MODEL EXPORT")
    print("=" * 60)

    # Load model
    from src.models import ModelFactory, ModelConfig

    print(f"\nLoading model from: {args.weights}")

    config = ModelConfig(
        name="h2_detector_export",
        architecture="yolov8n",
        num_classes=9,
        input_size=args.imgsz,
    )

    model = ModelFactory.create(config, str(args.weights))

    print(f"  Architecture: {config.architecture}")
    print(f"  Input size: {args.imgsz}x{args.imgsz}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        suffix_map = {
            "onnx": ".onnx",
            "torchscript": ".torchscript",
            "tflite": ".tflite",
            "coreml": ".mlmodel",
            "engine": ".engine",
        }
        output_path = args.weights.with_suffix(suffix_map[args.format])

    # Export
    print(f"\nExporting to {args.format} format...")
    print(f"  Output: {output_path}")
    print(f"  Half precision: {args.half}")
    print(f"  Dynamic batch: {args.dynamic}")

    try:
        model.export_model(str(output_path), format=args.format)
        print(f"\nExport successful!")
        print(f"Model saved to: {output_path}")

    except Exception as e:
        print(f"\nExport failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
