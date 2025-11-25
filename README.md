# Natural Hydrogen Seep Detection - Multi-Layer AI System

## Overview
Replication of Ginzburg et al. (2025) research: "Identification of Natural Hydrogen Seeps: Leveraging AI for Automated Classification of Sub-Circular Depressions"

This system uses deep learning to identify potential natural hydrogen seeps by classifying sub-circular depressions (SCDs) from satellite imagery.

## Architecture

### Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Detection & Data Acquisition                      │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐      │
│  │   Sentinel-2 │  │ Google Maps │  │  Manual/AI   │      │
│  │   Fetcher    │  │   Scraper   │  │  Detection   │      │
│  └──────────────┘  └─────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Classification (Core AI Model)                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │          YOLOv8 Classification Model             │      │
│  │  ┌────────────────────────────────────────┐     │      │
│  │  │  Input: 640x640 RGB Image              │     │      │
│  │  │  Output: 9 Classes                      │     │      │
│  │  │    - SCDs (H2-related)                  │     │      │
│  │  │    - Fairy circles, forts, farm circles │     │      │
│  │  │    - Flooded dunes, impact craters      │     │      │
│  │  │    - Karst, salt lakes, thermokarst     │     │      │
│  │  └────────────────────────────────────────┘     │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Post-Processing & Refinement                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Morphometric│  │  Geological  │  │   Spatial    │      │
│  │  Analysis   │  │   Context    │  │  Statistics  │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
h2_seep_detection/
├── README.md
├── requirements.txt
├── config/
│   ├── model_config.yaml
│   ├── data_config.yaml
│   └── training_config.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── sentinel2_fetcher.py      # Fetch Sentinel-2 multispectral data
│   │   ├── google_maps_scraper.py    # Scrape Google Maps imagery
│   │   ├── dataset_builder.py        # Build training datasets
│   │   └── augmentation.py           # Data augmentation pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yolo_classifier.py        # YOLOv8 wrapper
│   │   ├── ensemble.py               # Optional ensemble models
│   │   └── model_factory.py          # Factory pattern for models
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── spectral_indices.py       # NDVI, BI, etc.
│   │   ├── image_processor.py        # Image preprocessing
│   │   └── coordinate_handler.py     # GPS coordinate management
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   ├── morphometric.py           # Shape analysis
│   │   ├── geological_filter.py      # Geological context
│   │   └── spatial_stats.py          # Clustering analysis
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training orchestrator
│   │   ├── validator.py              # Validation logic
│   │   └── callbacks.py              # Training callbacks
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py              # Single image prediction
│   │   └── batch_predictor.py        # Batch processing
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── metrics.py
│       └── visualization.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_case_study_brazil.ipynb
├── tests/
│   └── ... (unit tests)
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
├── models/
│   └── weights/
└── outputs/
    └── predictions/
```

## Key Technical Details

### Data Pipeline
- **Sentinel-2:** 10m resolution, 13 spectral bands
- **Google Maps:** <10m resolution, RGB imagery
- **Indices:** NDVI (vegetation), BI (brightness)
- **Format:** Single structure per image, boundary-fitted

### Model Configuration
- **Architecture:** YOLOv8n/s/m (configurable)
- **Input Size:** 640x640 (standard YOLO)
- **Classes:** 9 (1 positive, 8 negative)
- **Training:** 50 epochs, AdamW optimizer
- **Augmentation:** Flip, rotate, brightness, contrast

### Performance Metrics
- Confusion Matrix (9x9)
- Precision/Recall per class
- F1-Score for SCD class
- ROC curves

## Design Patterns

1. **Factory Pattern:** Model creation and configuration
2. **Strategy Pattern:** Different data source strategies
3. **Pipeline Pattern:** Data preprocessing → Model → Post-processing
4. **Observer Pattern:** Training callbacks and monitoring
5. **Singleton Pattern:** Configuration management

## Functional Programming Principles
- Pure functions for transformations
- Immutable data structures where possible
- Function composition for pipelines
- Type hints throughout

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Configure project
cp config/model_config.yaml.example config/model_config.yaml
# Edit with your settings

# Download data
python src/data/sentinel2_fetcher.py --config config/data_config.yaml

# Train model
python src/training/trainer.py --config config/training_config.yaml

# Run inference
python src/inference/predictor.py --image path/to/image.jpg --model models/weights/best.pt
```

## References

Ginzburg, N., Daynac, J., Hesni, S., Geymond, U., & Roche, V. (2025). 
Identification of Natural Hydrogen Seeps: Leveraging AI for Automated Classification of Sub-Circular Depressions. 
Earth and Space Science, 12, e2025EA004227.
