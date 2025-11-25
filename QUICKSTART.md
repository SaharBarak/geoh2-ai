# Quick Start Guide - H2 Seep Detection

## Overview
This guide will help you get the H2 seep detection system running quickly.

---

## Prerequisites

### System Requirements
- **Python**: 3.9+
- **RAM**: 16GB minimum (32GB recommended)
- **GPU**: CUDA-compatible GPU recommended (RTX 3060 or better)
- **Storage**: 50GB+ for datasets and models

### API Keys (Optional for full functionality)
- Google Maps API key
- Sentinel Hub account
- Google Earth Engine account

---

## Installation

### Step 1: Clone or Extract Project
```bash
cd h2_seep_detection
```

### Step 2: Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n h2detect python=3.10
conda activate h2detect

# OR using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Example `.env`:
```bash
# Google Maps
GOOGLE_MAPS_API_KEY=your_key_here

# Sentinel Hub
SENTINEL_HUB_CLIENT_ID=your_client_id
SENTINEL_HUB_CLIENT_SECRET=your_secret

# Earth Engine
EARTHENGINE_ACCOUNT=your_account@gmail.com
```

---

## Quick Test Run

### Test 1: Model Initialization
```python
from src.models import ModelConfig
from src.models.yolo_classifier import YOLOv8Classifier

# Create configuration
config = ModelConfig(
    name="h2_detector",
    architecture="yolov8n",
    num_classes=9,
    class_names=["SCD", "fairy_circle", "fairy_fort", "farm_circle", 
                 "flooded_dune", "impact_crater", "karst", "salt_lake", "thermokarst"],
    input_size=640
)

# Initialize model
classifier = YOLOv8Classifier(config)
print("✓ Model initialized successfully!")
print(f"  Device: {classifier.device}")
print(f"  Parameters: {classifier.get_model_info()['parameters']:,}")
```

### Test 2: Spectral Indices
```python
from src.preprocessing.spectral_indices import SpectralIndexCalculator
import numpy as np

# Create sample bands
nir = np.random.rand(100, 100) * 0.6
red = np.random.rand(100, 100) * 0.4

# Calculate NDVI
calculator = SpectralIndexCalculator()
ndvi = calculator.ndvi(nir, red)

print(f"✓ NDVI calculated: range [{ndvi.value.min():.3f}, {ndvi.value.max():.3f}]")
```

---

## Usage Examples

### Example 1: Predict Single Image
```python
from src.models import ModelConfig
from src.models.yolo_classifier import YOLOv8Classifier

# Initialize
config = ModelConfig(
    name="h2_detector",
    architecture="yolov8s",
    num_classes=9,
    class_names=["SCD", "fairy_circle", "fairy_fort", "farm_circle", 
                 "flooded_dune", "impact_crater", "karst", "salt_lake", "thermokarst"],
    input_size=640,
    confidence_threshold=0.7
)

classifier = YOLOv8Classifier(
    config, 
    weights_path="models/weights/best.pt"  # Load pretrained weights
)

# Predict
result = classifier.predict("path/to/structure_image.jpg")

print(f"Class: {result.class_name}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Is SCD: {result.is_scd(threshold=0.7)}")

# Detailed probabilities
for class_name, prob in result.probabilities.items():
    print(f"  {class_name}: {prob:.2%}")
```

### Example 2: Batch Processing
```python
import glob

# Get all images in directory
image_paths = glob.glob("data/structures/*.jpg")

# Process in batches
results = classifier.predict_batch(image_paths, batch_size=16)

# Filter for high-confidence SCDs
scds = [r for r in results if r.is_scd(threshold=0.7)]

print(f"Found {len(scds)} potential H2 seeps out of {len(results)} structures")

# Save results
import json

with open("predictions.json", "w") as f:
    json.dump([r.to_dict() for r in results], f, indent=2)
```

### Example 3: Calculate Spectral Indices
```python
from src.preprocessing.spectral_indices import SpectralIndexCalculator
import rasterio

# Load Sentinel-2 bands
with rasterio.open("sentinel2_image.tif") as src:
    bands = {
        "B4": src.read(1),  # Red
        "B8": src.read(2),  # NIR
    }

# Calculate indices
calculator = SpectralIndexCalculator()

# Primary indices from paper
ndvi = calculator.compute("ndvi", bands)
bi = calculator.compute("bi", bands)

# Convert to 8-bit for visualization
ndvi_image = ndvi.to_uint8()
bi_image = bi.to_uint8()

# Save
import cv2
cv2.imwrite("ndvi.png", ndvi_image)
cv2.imwrite("bi.png", bi_image)
```

### Example 4: Training (if you have dataset)
```python
from src.models import ModelConfig
from src.models.yolo_classifier import YOLOv8Classifier

# Configure model
config = ModelConfig(
    name="h2_detector",
    architecture="yolov8m",  # Medium model
    num_classes=9,
    class_names=["SCD", "fairy_circle", "fairy_fort", "farm_circle", 
                 "flooded_dune", "impact_crater", "karst", "salt_lake", "thermokarst"],
    input_size=640
)

# Initialize
classifier = YOLOv8Classifier(config)

# Train
results = classifier.train_model(
    data_yaml="data/dataset.yaml",
    epochs=50,
    batch_size=16,
    learning_rate=0.001,
    patience=10,
    save_dir="runs/train"
)

print(f"Training complete!")
print(f"Best accuracy: {results.metrics['accuracy']:.2%}")
```

---

## Common Workflows

### Workflow 1: Analyzing New Region

```python
# 1. Define region of interest
roi_coords = [
    (-45.123, -15.456),  # (lon, lat)
    (-45.234, -15.567),
    # ... more coordinates
]

# 2. Download imagery
from src.data.google_maps_scraper import GoogleMapsScraper

scraper = GoogleMapsScraper()
images = scraper.download_batch(roi_coords, zoom=18)

# 3. Run predictions
results = classifier.predict_batch(images)

# 4. Filter and analyze
high_confidence_scds = [
    r for r in results 
    if r.is_scd(threshold=0.75)
]

# 5. Apply post-processing
from src.postprocessing.spatial_stats import cluster_analysis

clusters = cluster_analysis(high_confidence_scds, distance_threshold=5000)

# 6. Generate report
print(f"Analysis Results:")
print(f"  Total structures: {len(results)}")
print(f"  Potential H2 seeps: {len(high_confidence_scds)}")
print(f"  Clusters identified: {len(clusters)}")
```

### Workflow 2: Evaluating Model Performance

```python
from src.training.validator import Validator
from src.utils.metrics import confusion_matrix, classification_report

# Load test dataset
test_images = glob.glob("data/test/*.jpg")
test_labels = load_labels("data/test/labels.json")

# Run predictions
predictions = classifier.predict_batch(test_images)

# Calculate metrics
validator = Validator()
metrics = validator.evaluate(predictions, test_labels)

print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
print(f"SCD Precision: {metrics['scd_precision']:.2%}")
print(f"SCD Recall: {metrics['scd_recall']:.2%}")
print(f"F1-Score: {metrics['scd_f1']:.2%}")

# Confusion matrix
cm = confusion_matrix(predictions, test_labels)
print("\nConfusion Matrix:")
print(cm)
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or use smaller model
```python
config.architecture = "yolov8n"  # Use nano model
classifier.predict_batch(images, batch_size=4)  # Smaller batch
```

### Issue: API Rate Limits
**Solution**: Implement caching and delays
```python
from time import sleep

for coord in coordinates:
    image = scraper.download(coord)
    sleep(0.5)  # Delay between requests
```

### Issue: Low Accuracy
**Solution**: Try ensemble or adjust threshold
```python
# Use ensemble
from src.models import EnsembleModel

models = [
    YOLOv8Classifier(config),
    YOLOv8Classifier(config),
]
ensemble = EnsembleModel(models)

# Or adjust threshold
results = [r for r in predictions if r.confidence > 0.8]
```

---

## Next Steps

1. **Prepare Your Dataset**
   - Collect structure coordinates
   - Download imagery
   - Annotate data (if training)

2. **Train or Fine-tune Model**
   - Use provided training script
   - Monitor with TensorBoard
   - Validate on held-out data

3. **Deploy Model**
   - Export to ONNX
   - Set up API server
   - Create batch processing pipeline

4. **Integrate Post-processing**
   - Add morphometric filters
   - Include geological context
   - Perform spatial analysis

---

## Resources

### Documentation
- `README.md` - Project overview
- `ARCHITECTURE.md` - Detailed system design
- `examples/` - Code examples
- `notebooks/` - Jupyter tutorials

### Data Sources
- **Google Maps**: High-res imagery
- **Sentinel-2**: Multispectral data (free)
- **Training Data**: See Zenodo (doi:10.5281/zenodo.15376871)

### Community
- GitHub Issues - Bug reports
- Discussions - Questions and ideas
- Paper - Original research methodology

---

## Performance Benchmarks

### Inference Speed (single RTX 3060)
- **YOLOv8n**: ~50 images/sec
- **YOLOv8s**: ~40 images/sec
- **YOLOv8m**: ~25 images/sec

### Accuracy (on validation set)
- **Google Maps + YOLOv8**: 90%
- **Sentinel-2 + YOLOv8**: 70%
- **Ensemble**: 92%

---

## Citation

If you use this system in research, please cite:

```bibtex
@article{ginzburg2025hydrogen,
  title={Identification of Natural Hydrogen Seeps: Leveraging AI for Automated Classification of Sub-Circular Depressions},
  author={Ginzburg, N. and Daynac, J. and Hesni, S. and Geymond, U. and Roche, V.},
  journal={Earth and Space Science},
  volume={12},
  pages={e2025EA004227},
  year={2025},
  doi={10.1029/2025EA004227}
}
```
