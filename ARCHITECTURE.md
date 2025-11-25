# H2 Seep Detection - System Architecture

## Research Foundation

This implementation replicates the methodology from:

**Ginzburg, N., Daynac, J., Hesni, S., Geymond, U., & Roche, V. (2025)**  
"Identification of Natural Hydrogen Seeps: Leveraging AI for Automated Classification of Sub-Circular Depressions"  
*Earth and Space Science*, 12, e2025EA004227

### Key Findings from Paper
- **90% accuracy** using Google Maps high-resolution imagery
- **70% accuracy** using Sentinel-2 multispectral indices (NDVI + BI)
- Successfully screened **~2,000 structures** in Brazil's São Francisco Basin
- Discarded **52% as non-H2 structures**, focusing exploration efforts

---

## Architecture Overview

### Design Principles
1. **Modularity**: Each component is independent and replaceable
2. **Composability**: Components work together via clean interfaces
3. **Extensibility**: Easy to add new models, indices, or data sources
4. **Type Safety**: Full type hints throughout
5. **Functional Patterns**: Pure functions where possible
6. **OOP Patterns**: Factory, Strategy, Template Method, Observer

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   CLI Tool   │  │  Web API     │  │   Jupyter    │ │
│  │              │  │  (FastAPI)   │  │  Notebooks   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   Service Layer                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Pipeline Orchestrator                     │  │
│  │  (Coordinates data → model → postprocessing)     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   Core Modules                          │
│                                                         │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────┐ │
│  │  Data Module   │  │  Model Module  │  │  Utils   │ │
│  │                │  │                │  │          │ │
│  │ • Sentinel-2   │  │ • YOLOv8       │  │ • Logger │ │
│  │ • Google Maps  │  │ • Ensemble     │  │ • Metrics│ │
│  │ • Augmentation │  │ • Custom       │  │ • Viz    │ │
│  └────────────────┘  └────────────────┘  └──────────┘ │
│                                                         │
│  ┌────────────────────────┐  ┌────────────────────┐   │
│  │  Preprocessing Module  │  │ Postprocessing     │   │
│  │                        │  │                    │   │
│  │ • Spectral Indices     │  │ • Morphometric     │   │
│  │ • Image Processing     │  │ • Geological       │   │
│  │ • Coordinate Handling  │  │ • Spatial Stats    │   │
│  └────────────────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  PyTorch     │  │  Google EE   │  │  Storage     │ │
│  │  Ultralytics │  │  Sentinel Hub│  │  (Local/S3)  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Module Details

### 1. Data Module (`src/data/`)

#### Components:
- **Sentinel2Fetcher**: Retrieves multispectral satellite imagery
- **GoogleMapsScraper**: Downloads high-resolution visible imagery  
- **DatasetBuilder**: Constructs training/validation datasets
- **Augmentation**: Applies data augmentation strategies

#### Key Features:
- Automatic coordinate handling (WGS84)
- Cloud filtering for Sentinel-2
- Seasonal preference (dry season)
- Caching for performance

#### Data Flow:
```
Coordinates → API Query → Download → Cache → Preprocess → Dataset
```

---

### 2. Preprocessing Module (`src/preprocessing/`)

#### Components:
- **SpectralIndexCalculator**: Computes vegetation/brightness indices
- **ImageProcessor**: Handles image transformations
- **CoordinateHandler**: GPS coordinate conversions

#### Spectral Indices (Strategy Pattern):
| Index | Formula | Purpose |
|-------|---------|---------|
| NDVI | (NIR-RED)/(NIR+RED) | Vegetation anomalies (primary) |
| BI | sqrt((RED²+NIR²)/2) | Brightness/salt distinction (secondary) |
| NDWI | (GREEN-NIR)/(GREEN+NIR) | Water detection |
| SAVI | Soil-adjusted NDVI | Reduced soil influence |
| SI | sqrt(GREEN*RED) | Salinity detection |

#### Processing Pipeline:
```python
# Functional composition
process = compose(
    normalize,
    compute_indices,
    resize_to_target,
    apply_augmentation
)
result = process(raw_image)
```

---

### 3. Model Module (`src/models/`)

#### Class Hierarchy (Template Method Pattern):
```
BaseDetectionModel (ABC)
    ↓
├── YOLOv8Classifier (Concrete)
└── EnsembleModel (Composite)
```

#### YOLOv8Classifier Features:
- **Architecture Variants**: n/s/m/l/x (nano to extra-large)
- **Input**: 640×640 RGB images
- **Output**: 9-class probabilities
- **Methods**:
  - `predict()`: Single image inference
  - `predict_batch()`: Batch processing
  - `train_model()`: Training workflow
  - `validate_model()`: Validation metrics
  - `export_model()`: ONNX/TorchScript export

#### Model Factory Pattern:
```python
def create_model(config: dict) -> BaseDetectionModel:
    """Factory for model creation"""
    if config['type'] == 'yolov8':
        return YOLOv8Classifier(...)
    elif config['type'] == 'ensemble':
        return EnsembleModel(...)
    raise ValueError(f"Unknown model: {config['type']}")
```

---

### 4. Postprocessing Module (`src/postprocessing/`)

#### Components:
- **Morphometric Analysis**: Shape descriptors (area, circularity, depth)
- **Geological Filter**: Context from geological maps
- **Spatial Statistics**: Clustering and distribution analysis

#### Refinement Pipeline (Chain of Responsibility):
```
Raw Predictions
    ↓
[Confidence Filter] → threshold > 0.5
    ↓
[Morphometric Filter] → size, shape, depth criteria
    ↓
[Geological Filter] → sedimentary basin check
    ↓
[Spatial Filter] → cluster vs isolated
    ↓
Final Predictions
```

---

### 5. Training Module (`src/training/`)

#### Components:
- **Trainer**: Orchestrates training loop
- **Validator**: Computes validation metrics
- **Callbacks**: Early stopping, checkpointing, logging

#### Training Pipeline (Observer Pattern):
```python
trainer = Trainer(model, config)
trainer.add_callback(EarlyStoppingCallback(patience=10))
trainer.add_callback(CheckpointCallback(save_best=True))
trainer.add_callback(TensorBoardCallback())

results = trainer.train(
    train_loader=train_data,
    val_loader=val_data,
    epochs=50
)
```

---

## Data Structures

### Core Types:
```python
@dataclass
class PredictionResult:
    """Single prediction output"""
    class_name: str
    class_id: int
    confidence: float
    probabilities: Dict[str, float]
    image_path: Optional[str]
    metadata: Optional[Dict]

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    architecture: str
    num_classes: int
    class_names: List[str]
    input_size: int
    confidence_threshold: float
    device: str

@dataclass
class IndexResult:
    """Spectral index computation result"""
    name: str
    value: np.ndarray
    valid_range: Tuple[float, float]
    metadata: Dict
```

---

## Design Patterns Used

### 1. **Factory Pattern** (Model Creation)
```python
class ModelFactory:
    @staticmethod
    def create(config: ModelConfig) -> BaseDetectionModel:
        # Instantiate appropriate model based on config
```

### 2. **Strategy Pattern** (Spectral Indices)
```python
class SpectralIndexCalculator:
    def __init__(self):
        self._indices = {
            'ndvi': self.ndvi,
            'bi': self.brightness_index,
            # ... more strategies
        }
```

### 3. **Template Method** (Base Model)
```python
class BaseDetectionModel:
    def predict(self, image):
        # Template defines the algorithm
        tensor = self.preprocess(image)
        output = self.forward(tensor)
        return self.postprocess(output)
```

### 4. **Composite Pattern** (Ensemble)
```python
class EnsembleModel(BaseDetectionModel):
    def __init__(self, models: List[BaseDetectionModel]):
        self.models = models  # Composite of models
```

### 5. **Observer Pattern** (Training Callbacks)
```python
class Trainer:
    def add_callback(self, callback: Callback):
        self.callbacks.append(callback)
    
    def notify(self, event: str, data: Dict):
        for callback in self.callbacks:
            callback.on_event(event, data)
```

### 6. **Singleton Pattern** (Config Manager)
```python
class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

---

## Functional Programming Elements

### Pure Functions:
```python
def normalize_image(image: np.ndarray) -> np.ndarray:
    """Pure function: no side effects"""
    return (image - image.min()) / (image.max() - image.min())

def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Pure function: deterministic"""
    return (nir - red) / (nir + red + 1e-8)
```

### Function Composition:
```python
from functools import reduce

def compose(*functions):
    """Compose functions right-to-left"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

# Usage
preprocess = compose(
    normalize,
    resize(640),
    to_tensor
)
```

### Immutable Data:
```python
from dataclasses import dataclass, FrozenInstanceError

@dataclass(frozen=True)
class ImmutableConfig:
    """Immutable configuration"""
    learning_rate: float
    batch_size: int
```

---

## Performance Optimizations

### 1. **Batch Processing**
- Vectorized operations with NumPy
- GPU batch inference
- Parallel data loading

### 2. **Caching**
- Disk cache for downloaded imagery
- Memory cache for preprocessed data
- Model output caching

### 3. **Mixed Precision Training**
- FP16 for faster training
- Automatic mixed precision (AMP)

### 4. **Data Pipeline**
- Async I/O for downloads
- Prefetching with DataLoader
- Memory-mapped arrays for large datasets

---

## Extension Points

### Adding New Data Sources:
```python
class CustomDataSource(BaseDataSource):
    def fetch(self, coordinates: Tuple[float, float]) -> np.ndarray:
        # Implement custom data fetching
        pass
```

### Adding New Models:
```python
class CustomModel(BaseDetectionModel):
    def build_model(self) -> nn.Module:
        # Implement custom architecture
        pass
```

### Adding New Indices:
```python
calculator = SpectralIndexCalculator()
calculator._indices['custom'] = lambda **bands: custom_index(**bands)
```

---

## Testing Strategy

### Unit Tests:
- Test each module independently
- Mock external dependencies
- Property-based testing for mathematical functions

### Integration Tests:
- Test complete pipeline
- Test with sample data
- Test error handling

### Performance Tests:
- Benchmark inference speed
- Memory profiling
- Scalability testing

---

## Deployment Options

### 1. **Local Development**
```bash
python src/inference/predictor.py --image path/to/image.jpg
```

### 2. **API Server** (FastAPI)
```python
@app.post("/predict")
async def predict(file: UploadFile):
    result = classifier.predict(file)
    return result.to_dict()
```

### 3. **Batch Processing** (Cloud)
```bash
# Process large dataset on cloud
python scripts/batch_process.py --input s3://bucket/images/
```

### 4. **Model Export**
```python
# Export to ONNX for deployment
classifier.export_model(format='onnx', output='model.onnx')
```

---

## References

1. Ginzburg et al. (2025) - Original research methodology
2. Ultralytics YOLOv8 - Detection framework
3. Sentinel-2 - Multispectral satellite data
4. Tucker (1979) - NDVI formulation
5. Design Patterns - Gang of Four (GoF)
