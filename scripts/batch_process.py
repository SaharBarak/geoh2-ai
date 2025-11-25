#!/usr/bin/env python3
"""
Batch Processing Script

Process large numbers of images for H2 seep detection.
Supports parallel processing and result aggregation.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def process_images(
    image_paths: List[str],
    model,
    batch_size: int = 16,
    threshold: float = 0.5,
    show_progress: bool = True,
) -> Tuple[List, dict]:
    """
    Process images and return results with summary.

    Args:
        image_paths: List of image file paths
        model: Detection model
        batch_size: Batch size for processing
        threshold: Confidence threshold for SCD
        show_progress: Show progress bar

    Returns:
        Tuple of (results list, summary dict)
    """
    from collections import Counter

    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(image_paths), batch_size), desc="Processing")
        except ImportError:
            iterator = range(0, len(image_paths), batch_size)
    else:
        iterator = range(0, len(image_paths), batch_size)

    all_results = []

    for i in iterator:
        batch = image_paths[i:i + batch_size]
        results = model.predict_batch(batch)

        for path, result in zip(batch, results):
            all_results.append({
                "image": path,
                "class_name": result.class_name,
                "class_id": result.class_id,
                "confidence": result.confidence,
                "is_scd": result.is_scd(threshold),
                "probabilities": result.probabilities,
            })

    # Generate summary
    class_counts = Counter(r["class_name"] for r in all_results)
    scd_count = sum(1 for r in all_results if r["is_scd"])
    high_conf_scds = [r for r in all_results if r["is_scd"] and r["confidence"] > 0.7]

    summary = {
        "total_processed": len(all_results),
        "scd_count": scd_count,
        "scd_percentage": 100 * scd_count / len(all_results) if all_results else 0,
        "high_confidence_scds": len(high_conf_scds),
        "class_distribution": dict(class_counts),
        "threshold": threshold,
        "timestamp": datetime.now().isoformat(),
    }

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch process images for H2 seep detection"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default="batch_results.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--weights", "-w",
        help="Path to model weights"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Confidence threshold for SCD classification"
    )
    parser.add_argument(
        "--extensions",
        default="jpg,jpeg,png,tif,tiff",
        help="Image file extensions (comma-separated)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search directories recursively"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also output CSV file"
    )

    args = parser.parse_args()

    # Find images
    print(f"Searching for images in: {args.input}")

    image_paths = []
    extensions = args.extensions.split(",")

    for ext in extensions:
        pattern = f"**/*.{ext}" if args.recursive else f"*.{ext}"
        image_paths.extend(glob.glob(str(args.input / pattern), recursive=args.recursive))

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")

    # Load model
    print("Loading model...")
    from src.models import ModelFactory

    if args.config:
        model = ModelFactory.create_from_yaml(args.config)
    else:
        model = ModelFactory.create_default(args.weights)

    print(f"Model: {model.config.name}")
    print(f"Device: {model.device}")

    # Process
    print("\nProcessing images...")
    results, summary = process_images(
        image_paths,
        model,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total processed:        {summary['total_processed']}")
    print(f"Potential H2 seeps:     {summary['scd_count']} ({summary['scd_percentage']:.1f}%)")
    print(f"High-confidence SCDs:   {summary['high_confidence_scds']}")
    print(f"\nClass Distribution:")
    for class_name, count in sorted(summary["class_distribution"].items(), key=lambda x: -x[1]):
        pct = 100 * count / summary["total_processed"]
        print(f"  {class_name:20} {count:5} ({pct:5.1f}%)")

    # Save results
    output_data = {
        "summary": summary,
        "results": results,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Save CSV if requested
    if args.csv:
        csv_path = args.output.with_suffix(".csv")
        import csv

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image", "class_name", "class_id", "confidence", "is_scd"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "image": r["image"],
                    "class_name": r["class_name"],
                    "class_id": r["class_id"],
                    "confidence": f"{r['confidence']:.4f}",
                    "is_scd": r["is_scd"],
                })
        print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
