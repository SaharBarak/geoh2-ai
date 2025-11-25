#!/usr/bin/env python3
"""
Imagery Fetching Script

Download satellite imagery for specified coordinates.
Supports Google Maps and Sentinel-2 sources.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_coordinates(input_path: Path) -> List[Tuple[float, float, str]]:
    """
    Load coordinates from file.

    Supports CSV and JSON formats.

    Returns:
        List of (lon, lat, name) tuples
    """
    coords = []

    if input_path.suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)
            for item in data:
                if isinstance(item, dict):
                    coords.append((
                        item.get("longitude", item.get("lon")),
                        item.get("latitude", item.get("lat")),
                        item.get("name", f"point_{len(coords)}"),
                    ))
                elif isinstance(item, (list, tuple)):
                    coords.append((item[0], item[1], f"point_{len(coords)}"))

    elif input_path.suffix == ".csv":
        with open(input_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                lon = float(row.get("longitude", row.get("lon", row.get("x"))))
                lat = float(row.get("latitude", row.get("lat", row.get("y"))))
                name = row.get("name", row.get("id", f"point_{len(coords)}"))
                coords.append((lon, lat, name))

    else:
        # Try plain text: lon,lat per line
        with open(input_path) as f:
            for i, line in enumerate(f):
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1]), f"point_{i}"))

    return coords


def main():
    parser = argparse.ArgumentParser(
        description="Fetch satellite imagery for coordinates"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input file with coordinates (CSV, JSON, or text)"
    )
    parser.add_argument(
        "--lon",
        type=float,
        help="Single longitude"
    )
    parser.add_argument(
        "--lat",
        type=float,
        help="Single latitude"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for images"
    )
    parser.add_argument(
        "--source", "-s",
        default="google_maps",
        choices=["google_maps", "sentinel2"],
        help="Imagery source"
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=18,
        help="Zoom level for Google Maps"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Image size in pixels"
    )
    parser.add_argument(
        "--width",
        type=float,
        default=640,
        help="Area width in meters (for Sentinel-2)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("H2 SEEP DETECTION - IMAGERY FETCHER")
    print("=" * 60)

    # Get coordinates
    if args.input:
        print(f"\nLoading coordinates from: {args.input}")
        coords = load_coordinates(args.input)
    elif args.lon is not None and args.lat is not None:
        coords = [(args.lon, args.lat, "single_point")]
    else:
        print("Error: Provide either --input or --lon/--lat")
        sys.exit(1)

    print(f"Found {len(coords)} coordinates")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize fetcher
    print(f"\nSource: {args.source}")

    if args.source == "google_maps":
        from src.data import GoogleMapsScraper, GoogleMapsConfig

        config = GoogleMapsConfig(
            zoom_level=args.zoom,
            image_size=args.size,
        )
        fetcher = GoogleMapsScraper(config)

        print(f"  Zoom: {args.zoom}")
        print(f"  Size: {args.size}x{args.size}")

    else:
        from src.data import Sentinel2Fetcher, Sentinel2Config

        config = Sentinel2Config()
        fetcher = Sentinel2Fetcher(config)

        print(f"  Width: {args.width}m")

    # Fetch images
    print(f"\nFetching imagery...")

    results = []
    try:
        from tqdm import tqdm
        iterator = tqdm(coords)
    except ImportError:
        iterator = coords

    for lon, lat, name in iterator:
        try:
            if args.source == "google_maps":
                result = fetcher.download(lon, lat, zoom=args.zoom, size=args.size)
                if result:
                    # Save image
                    import cv2
                    output_file = args.output / f"{name}.jpg"
                    cv2.imwrite(str(output_file), result.image[:, :, ::-1])  # RGB to BGR
                    results.append({
                        "name": name,
                        "lon": lon,
                        "lat": lat,
                        "file": str(output_file),
                        "success": True,
                    })
                else:
                    results.append({
                        "name": name,
                        "lon": lon,
                        "lat": lat,
                        "success": False,
                        "error": "Failed to fetch",
                    })

            else:  # sentinel2
                result = fetcher.fetch(lon, lat, width_meters=args.width)
                if result:
                    # Save RGB image
                    import cv2
                    rgb = result.get_rgb()
                    output_file = args.output / f"{name}.jpg"
                    cv2.imwrite(str(output_file), rgb[:, :, ::-1])
                    results.append({
                        "name": name,
                        "lon": lon,
                        "lat": lat,
                        "file": str(output_file),
                        "success": True,
                    })
                else:
                    results.append({
                        "name": name,
                        "lon": lon,
                        "lat": lat,
                        "success": False,
                        "error": "No data available",
                    })

        except Exception as e:
            results.append({
                "name": name,
                "lon": lon,
                "lat": lat,
                "success": False,
                "error": str(e),
            })

    # Summary
    success_count = sum(1 for r in results if r["success"])

    print(f"\n" + "=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    print(f"Total coordinates: {len(coords)}")
    print(f"Successfully fetched: {success_count}")
    print(f"Failed: {len(coords) - success_count}")
    print(f"Output directory: {args.output}")

    # Save manifest
    manifest_path = args.output / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Manifest saved to: {manifest_path}")

    # Cleanup
    if hasattr(fetcher, 'close'):
        fetcher.close()


if __name__ == "__main__":
    main()
