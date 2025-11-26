"""
Integration Tests - Complete Pipeline

Tests the full workflow from image loading to prediction.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestModelPipeline:
    """Test model loading and prediction pipeline."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple RGB image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Save to temp file
        import cv2
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            yield f.name

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        from src.models import ModelConfig

        return ModelConfig(
            name="test_model",
            architecture="yolov8n",
            num_classes=9,
            class_names=[
                "SCD", "fairy_circle", "fairy_fort", "farm_circle",
                "flooded_dune", "impact_crater", "karst", "salt_lake",
                "thermokarst"
            ],
            input_size=640,
            confidence_threshold=0.5,
            device="cpu",  # Use CPU for testing
        )

    def test_model_factory_creation(self, model_config):
        """Test model creation via factory."""
        from src.models import ModelFactory

        model = ModelFactory.create(model_config)

        assert model is not None
        assert model.config.name == "test_model"
        assert model.config.num_classes == 9

    def test_model_prediction_format(self, model_config, sample_image):
        """Test that prediction returns correct format."""
        from src.models import ModelFactory, PredictionResult

        model = ModelFactory.create(model_config)
        result = model.predict(sample_image)

        # Check result type
        assert isinstance(result, PredictionResult)

        # Check required fields
        assert hasattr(result, "class_name")
        assert hasattr(result, "class_id")
        assert hasattr(result, "confidence")
        assert hasattr(result, "probabilities")

        # Check confidence is valid
        assert 0 <= result.confidence <= 1

        # Check probabilities sum to ~1
        prob_sum = sum(result.probabilities.values())
        assert 0.99 <= prob_sum <= 1.01

    def test_batch_prediction(self, model_config, sample_image):
        """Test batch prediction."""
        from src.models import ModelFactory

        model = ModelFactory.create(model_config)

        # Create multiple images
        images = [sample_image] * 3
        results = model.predict_batch(images, batch_size=2)

        assert len(results) == 3
        for result in results:
            assert 0 <= result.confidence <= 1

    def test_prediction_result_methods(self, model_config, sample_image):
        """Test PredictionResult helper methods."""
        from src.models import ModelFactory

        model = ModelFactory.create(model_config)
        result = model.predict(sample_image)

        # Test is_scd method
        is_scd = result.is_scd(threshold=0.5)
        assert isinstance(is_scd, bool)

        # Test to_dict method
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "class_name" in result_dict
        assert "confidence" in result_dict


class TestSpectralIndicesPipeline:
    """Test spectral index calculation pipeline."""

    @pytest.fixture
    def sample_bands(self):
        """Create sample Sentinel-2 bands."""
        np.random.seed(42)
        height, width = 100, 100

        return {
            "B2": np.random.rand(height, width).astype(np.float32) * 0.1,
            "B3": np.random.rand(height, width).astype(np.float32) * 0.1,
            "B4": np.random.rand(height, width).astype(np.float32) * 0.15,
            "B8": np.random.rand(height, width).astype(np.float32) * 0.4,
            "B11": np.random.rand(height, width).astype(np.float32) * 0.2,
            "B12": np.random.rand(height, width).astype(np.float32) * 0.15,
        }

    def test_ndvi_calculation(self, sample_bands):
        """Test NDVI calculation."""
        from src.preprocessing import SpectralIndexCalculator

        calculator = SpectralIndexCalculator()
        result = calculator.ndvi(bands=sample_bands)

        assert result is not None
        assert result.name == "NDVI"
        assert result.valid_range == (-1, 1)
        assert result.value.shape == (100, 100)

        # Check values are in valid range
        assert np.nanmin(result.value) >= -1
        assert np.nanmax(result.value) <= 1

    def test_brightness_index_calculation(self, sample_bands):
        """Test Brightness Index calculation."""
        from src.preprocessing import SpectralIndexCalculator

        calculator = SpectralIndexCalculator()
        result = calculator.brightness_index(bands=sample_bands)

        assert result is not None
        assert result.name == "BI"
        assert result.valid_range == (0, 1)

    def test_multiple_indices(self, sample_bands):
        """Test computing multiple indices."""
        from src.preprocessing import SpectralIndexCalculator

        calculator = SpectralIndexCalculator()
        indices = ["ndvi", "bi", "ndwi", "savi"]
        results = calculator.compute_multiple(indices, sample_bands)

        assert len(results) == 4
        assert "ndvi" in results
        assert "bi" in results

    def test_index_to_uint8(self, sample_bands):
        """Test conversion to 8-bit for visualization."""
        from src.preprocessing import SpectralIndexCalculator

        calculator = SpectralIndexCalculator()
        result = calculator.ndvi(bands=sample_bands)

        uint8_image = result.to_uint8()
        assert uint8_image.dtype == np.uint8
        assert uint8_image.min() >= 0
        assert uint8_image.max() <= 255


class TestImageProcessingPipeline:
    """Test image preprocessing pipeline."""

    @pytest.fixture
    def sample_image_array(self):
        """Create sample image array."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_image_processor_resize(self, sample_image_array):
        """Test image resizing."""
        from src.preprocessing import ImageProcessor

        processor = ImageProcessor(target_size=(640, 640))
        result = processor.resize(sample_image_array)

        assert result.shape == (640, 640, 3)

    def test_image_processor_normalize(self, sample_image_array):
        """Test image normalization."""
        from src.preprocessing import ImageProcessor

        processor = ImageProcessor()
        result = processor.normalize_image(sample_image_array)

        assert result.dtype == np.float32
        assert result.min() >= 0
        assert result.max() <= 1

    def test_image_processor_pipeline(self, sample_image_array):
        """Test complete preprocessing pipeline."""
        from src.preprocessing import ImageProcessor

        processor = ImageProcessor(target_size=(640, 640), normalize=True)
        result = processor.process(sample_image_array)

        assert result.shape == (640, 640, 3)
        assert result.dtype == np.float32

    def test_tensor_format_conversion(self, sample_image_array):
        """Test HWC to CHW conversion."""
        from src.preprocessing import ImageProcessor

        processor = ImageProcessor()
        tensor = processor.to_tensor_format(sample_image_array)

        assert tensor.shape == (3, 480, 640)  # CHW format


class TestCoordinateHandling:
    """Test coordinate transformation pipeline."""

    def test_bounding_box_creation(self):
        """Test creating bounding box from coordinates."""
        from src.preprocessing import CoordinateHandler

        handler = CoordinateHandler()
        bbox = handler.create_bounding_box(
            center_lon=-44.5,
            center_lat=-15.5,
            width_meters=640,
        )

        assert bbox is not None
        assert bbox.center == pytest.approx((-44.5, -15.5), rel=0.01)

    def test_haversine_distance(self):
        """Test distance calculation."""
        from src.preprocessing import CoordinateHandler

        handler = CoordinateHandler()

        # Distance from São Paulo to Rio de Janeiro (~360 km)
        distance = handler.haversine_distance(
            -46.6333, -23.5505,  # São Paulo
            -43.1729, -22.9068,  # Rio de Janeiro
        )

        # Should be approximately 360 km
        assert 350000 < distance < 400000  # meters

    def test_coordinate_validation(self):
        """Test coordinate validation."""
        from src.preprocessing import CoordinateHandler

        handler = CoordinateHandler()

        assert handler.validate_coordinates(-44.5, -15.5)
        assert handler.validate_coordinates(180, 90)
        assert not handler.validate_coordinates(181, 0)
        assert not handler.validate_coordinates(0, 91)


class TestPostProcessingPipeline:
    """Test post-processing pipeline."""

    def test_known_h2_fields_checker(self):
        """Test proximity to known H2 fields."""
        from src.postprocessing import KnownH2FieldsChecker

        checker = KnownH2FieldsChecker()

        # Test location near São Francisco Basin
        is_near, field_name, distance = checker.check(-44.5, -15.5)
        assert is_near
        assert field_name is not None

        # Test location far from any known field
        is_near, field_name, distance = checker.check(0, 0)
        assert not is_near


class TestTrainingPipeline:
    """Test training infrastructure."""

    def test_training_config_creation(self):
        """Test training configuration."""
        from src.training import TrainingConfig

        config = TrainingConfig(
            epochs=10,
            batch_size=8,
            learning_rate=0.001,
        )

        assert config.epochs == 10
        assert config.batch_size == 8
        assert config.learning_rate == 0.001

    def test_validator_metrics(self):
        """Test validation metrics computation."""
        from src.training import Validator

        validator = Validator(class_names=["class_a", "class_b", "class_c"])

        # Simulate predictions and ground truth
        predictions = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        ground_truth = [0, 1, 2, 0, 1, 1, 0, 2, 2, 1]

        result = validator.evaluate(predictions, ground_truth)

        assert 0 <= result.accuracy <= 1
        assert "class_a" in result.precision
        assert result.confusion_matrix.shape == (3, 3)

    def test_early_stopping_callback(self):
        """Test early stopping callback."""
        from src.training import EarlyStoppingCallback

        callback = EarlyStoppingCallback(patience=3, monitor="val_accuracy")

        # Simulate epochs with no improvement
        callback.on_train_begin({})
        callback.on_epoch_end({"val_accuracy": 0.5})
        assert not callback.should_stop

        callback.on_epoch_end({"val_accuracy": 0.5})
        callback.on_epoch_end({"val_accuracy": 0.5})
        callback.on_epoch_end({"val_accuracy": 0.5})

        assert callback.should_stop


class TestAPIEndpoints:
    """Test API endpoints."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.api import app

        return TestClient(app)

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_classes_endpoint(self, test_client):
        """Test classes endpoint."""
        response = test_client.get("/api/v1/classes")

        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert len(data["classes"]) == 9


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_complete_prediction_workflow(self, temp_dir):
        """Test complete workflow from image to prediction."""
        import cv2
        from src.models import ModelFactory, ModelConfig

        # Create test image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img_path = temp_dir / "test_image.jpg"
        cv2.imwrite(str(img_path), img)

        # Create model
        config = ModelConfig(
            name="e2e_test",
            architecture="yolov8n",
            num_classes=9,
            device="cpu",
        )
        model = ModelFactory.create(config)

        # Run prediction
        result = model.predict(str(img_path))

        # Verify result - Note: without custom trained weights, model may return
        # ImageNet classes instead of our 9 custom classes
        assert result.class_name is not None
        assert isinstance(result.class_id, int)
        assert 0 <= result.confidence <= 1

        # Save result
        result_path = temp_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f)

        # Verify saved result
        with open(result_path) as f:
            saved_result = json.load(f)
        assert saved_result["class_name"] == result.class_name

    def test_spectral_to_classification_pipeline(self, temp_dir):
        """Test pipeline from spectral indices to classification."""
        from src.preprocessing import SpectralIndexCalculator, ImageProcessor
        from src.models import ModelFactory, ModelConfig
        import cv2

        # Generate synthetic spectral data
        np.random.seed(42)
        bands = {
            "B4": np.random.rand(100, 100).astype(np.float32) * 0.15,
            "B8": np.random.rand(100, 100).astype(np.float32) * 0.4,
        }

        # Calculate NDVI
        calculator = SpectralIndexCalculator()
        ndvi = calculator.ndvi(bands=bands)

        # Convert to image (visualization)
        ndvi_image = ndvi.to_uint8()
        ndvi_rgb = np.stack([ndvi_image] * 3, axis=-1)

        # Resize for model
        processor = ImageProcessor(target_size=(640, 640))
        processed = processor.resize(ndvi_rgb)

        # Save and predict
        img_path = temp_dir / "ndvi.jpg"
        cv2.imwrite(str(img_path), processed)

        config = ModelConfig(
            name="spectral_test",
            architecture="yolov8n",
            num_classes=9,
            device="cpu",
        )
        model = ModelFactory.create(config)
        result = model.predict(str(img_path))

        assert result is not None
        assert result.class_name is not None
