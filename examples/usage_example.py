#!/usr/bin/env python3
"""
H2 Seep Detection - Usage Examples

This script demonstrates the main functionality of the H2 seep
detection system based on Ginzburg et al. (2025).

Examples:
    1. Single image prediction
    2. Batch processing
    3. Spectral index calculation
    4. Post-processing with geological context
    5. Training a model
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_single_prediction():
    """
    Example 1: Predict a single image.

    This is the simplest use case - classify one image
    as a potential H2 seep (SCD) or one of 8 other classes.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single Image Prediction")
    print("=" * 60)

    from src.models import ModelFactory, ModelConfig

    # Create model configuration
    config = ModelConfig(
        name="h2_detector",
        architecture="yolov8n",  # nano model for fast inference
        num_classes=9,
        class_names=[
            "SCD",           # Sub-Circular Depression (H2-related)
            "fairy_circle",  # Namibian fairy circles
            "fairy_fort",    # Irish ring forts
            "farm_circle",   # Center-pivot irrigation
            "flooded_dune",  # Flooded interdune areas
            "impact_crater", # Meteorite impacts
            "karst",         # Karst sinkholes
            "salt_lake",     # Circular salt lakes
            "thermokarst",   # Permafrost thaw lakes
        ],
        input_size=640,
        confidence_threshold=0.5,
    )

    # Initialize model (without pretrained weights for demo)
    model = ModelFactory.create(config)

    print(f"Model: {model.config.name}")
    print(f"Architecture: {model.config.architecture}")
    print(f"Device: {model.device}")
    print(f"Classes: {model.config.num_classes}")

    # Predict (would need actual image)
    # result = model.predict("path/to/structure_image.jpg")
    # print(f"Prediction: {result.class_name} ({result.confidence:.2%})")
    # print(f"Is H2 Seep: {result.is_scd()}")


def example_batch_processing():
    """
    Example 2: Batch processing multiple images.

    Process a directory of images efficiently using batching.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 60)

    from src.inference import H2SeepPredictor

    # Create predictor with default config
    # predictor = H2SeepPredictor.default()

    # Process batch
    # image_paths = glob.glob("data/structures/*.jpg")
    # results = predictor.predict_batch(image_paths, batch_size=16)

    # Analyze results
    # scd_count = sum(1 for r in results if r["is_scd"])
    # print(f"Total: {len(results)}, SCDs: {scd_count}")

    print("Batch processing would process multiple images efficiently")
    print("Use: predictor.predict_batch(image_paths, batch_size=16)")


def example_spectral_indices():
    """
    Example 3: Calculate spectral indices from Sentinel-2 data.

    NDVI and BI are the primary indices for H2 seep detection
    achieving 70% accuracy on Sentinel-2 imagery.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Spectral Index Calculation")
    print("=" * 60)

    import numpy as np
    from src.preprocessing import SpectralIndexCalculator

    # Create calculator
    calculator = SpectralIndexCalculator()

    print(f"Available indices: {calculator.available_indices}")

    # Simulate Sentinel-2 bands (in practice, load from GeoTIFF)
    # Using random data for demonstration
    np.random.seed(42)
    height, width = 100, 100

    bands = {
        "B2": np.random.rand(height, width) * 0.1,   # Blue
        "B3": np.random.rand(height, width) * 0.1,   # Green
        "B4": np.random.rand(height, width) * 0.15,  # Red
        "B8": np.random.rand(height, width) * 0.4,   # NIR
        "B11": np.random.rand(height, width) * 0.2,  # SWIR1
        "B12": np.random.rand(height, width) * 0.15, # SWIR2
    }

    # Calculate NDVI (primary index)
    ndvi = calculator.ndvi(bands=bands)
    print(f"\nNDVI (Normalized Difference Vegetation Index):")
    print(f"  Formula: (NIR - RED) / (NIR + RED)")
    print(f"  Mean: {np.nanmean(ndvi.value):.4f}")
    print(f"  Range: [{np.nanmin(ndvi.value):.4f}, {np.nanmax(ndvi.value):.4f}]")
    print(f"  Valid range: {ndvi.valid_range}")

    # Calculate BI (secondary index)
    bi = calculator.brightness_index(bands=bands)
    print(f"\nBI (Brightness Index):")
    print(f"  Formula: sqrt((RED^2 + NIR^2) / 2)")
    print(f"  Mean: {np.nanmean(bi.value):.4f}")
    print(f"  Range: [{np.nanmin(bi.value):.4f}, {np.nanmax(bi.value):.4f}]")

    # Calculate multiple indices at once
    indices = calculator.compute_multiple(["ndvi", "bi", "ndwi", "savi"], bands)
    print(f"\nComputed {len(indices)} indices: {list(indices.keys())}")


def example_geological_filtering():
    """
    Example 4: Apply geological context filtering.

    Refine predictions based on geological favorability:
    - Sedimentary basins
    - Proximity to faults
    - Known H2 fields
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Geological Context Filtering")
    print("=" * 60)

    from src.postprocessing import GeologicalFilter, KnownH2FieldsChecker

    # Check proximity to known H2 fields
    checker = KnownH2FieldsChecker()

    # Test coordinates (São Francisco Basin, Brazil)
    test_locations = [
        (-44.5, -15.5, "São Francisco Basin"),
        (-70.0, -20.0, "Chile - no known H2"),
        (-5.5, 13.5, "Mali - Bourakébougou"),
    ]

    print("\nKnown H2 Field Proximity Check:")
    for lon, lat, name in test_locations:
        is_near, field_name, distance = checker.check(lon, lat)
        if is_near:
            print(f"  {name}: Near '{field_name}' ({distance:.1f} km)")
        else:
            print(f"  {name}: Not near known H2 field")

    # Geological filter would use shapefiles
    # filter = GeologicalFilter(
    #     basins_shapefile="data/geo/sedimentary_basins.shp",
    #     faults_shapefile="data/geo/fault_lines.shp",
    # )
    # result = filter.filter(prediction, lon, lat)


def example_data_acquisition():
    """
    Example 5: Fetch satellite imagery for coordinates.

    Download imagery from Google Maps or Sentinel-2.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Data Acquisition")
    print("=" * 60)

    from src.data import GoogleMapsConfig, Sentinel2Config

    # Google Maps configuration
    gm_config = GoogleMapsConfig(
        zoom_level=18,       # High detail
        image_size=640,      # Standard size
        map_type="satellite",
    )
    print(f"\nGoogle Maps Config:")
    print(f"  Zoom: {gm_config.zoom_level}")
    print(f"  Size: {gm_config.image_size}x{gm_config.image_size}")

    # Sentinel-2 configuration
    s2_config = Sentinel2Config(
        resolution=10,        # 10m resolution
        max_cloud_cover=20,   # Max 20% clouds
        bands=["B2", "B3", "B4", "B8", "B11", "B12"],
    )
    print(f"\nSentinel-2 Config:")
    print(f"  Resolution: {s2_config.resolution}m")
    print(f"  Max cloud cover: {s2_config.max_cloud_cover}%")
    print(f"  Bands: {s2_config.bands}")

    # Usage (requires API keys):
    # scraper = GoogleMapsScraper(gm_config)
    # image = scraper.download(lon=-44.5, lat=-15.5)

    # fetcher = Sentinel2Fetcher(s2_config)
    # image = fetcher.fetch(lon=-44.5, lat=-15.5, width_meters=640)


def example_training():
    """
    Example 6: Training a model.

    Configure and train a YOLOv8 classifier.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Model Training")
    print("=" * 60)

    from src.training import TrainingConfig

    # Training configuration
    config = TrainingConfig(
        epochs=50,
        batch_size=16,
        learning_rate=0.001,
        patience=10,  # Early stopping
        optimizer="adamw",
        amp=True,     # Mixed precision
        save_dir="runs/train",
        seed=42,
    )

    print(f"Training Configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Early stopping patience: {config.patience}")
    print(f"  Mixed precision: {config.amp}")

    # Training (requires dataset):
    # from src.training import Trainer
    # from src.models import ModelFactory, ModelConfig
    #
    # model = ModelFactory.create_yolov8(size="n")
    # trainer = Trainer(model, config)
    # result = trainer.train(data_yaml="data/dataset.yaml")
    # print(f"Best accuracy: {result.best_accuracy:.2%}")


def example_complete_pipeline():
    """
    Example 7: Complete detection pipeline.

    End-to-end workflow from coordinates to filtered predictions.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Complete Detection Pipeline")
    print("=" * 60)

    print("""
    Complete Pipeline Steps:

    1. Define region of interest (coordinates)
    2. Download satellite imagery (Google Maps or Sentinel-2)
    3. Run classification model (YOLOv8)
    4. Apply post-processing filters:
       - Confidence threshold (>50%)
       - Morphometric analysis (size, shape)
       - Geological context (basin, faults)
       - Spatial clustering
    5. Generate report with visualizations

    Expected Results (per paper):
    - Google Maps: 90% accuracy
    - Sentinel-2: 70% accuracy
    - ~48% of structures classified as potential SCDs
    - Clustering in northern São Francisco Basin
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("H2 SEEP DETECTION - USAGE EXAMPLES")
    print("Based on Ginzburg et al. (2025)")
    print("=" * 60)

    example_single_prediction()
    example_batch_processing()
    example_spectral_indices()
    example_geological_filtering()
    example_data_acquisition()
    example_training()
    example_complete_pipeline()

    print("\n" + "=" * 60)
    print("For more information, see:")
    print("  - README.md")
    print("  - QUICKSTART.md")
    print("  - ARCHITECTURE.md")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
