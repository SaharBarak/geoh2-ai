"""
Pytest Configuration and Fixtures

Shared fixtures for all tests.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_dir(project_root):
    """Get config directory."""
    return project_root / "config"


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Get data directory."""
    return project_root / "data"


@pytest.fixture
def sample_config():
    """Create sample model configuration."""
    from src.models import ModelConfig

    return ModelConfig(
        name="test_model",
        architecture="yolov8n",
        num_classes=9,
        class_names=[
            "SCD",
            "fairy_circle",
            "fairy_fort",
            "farm_circle",
            "flooded_dune",
            "impact_crater",
            "karst",
            "salt_lake",
            "thermokarst",
        ],
        input_size=640,
        confidence_threshold=0.5,
        device="cpu",
    )


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample test image and return its path."""
    import cv2
    import numpy as np

    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img)

    return img_path


@pytest.fixture
def sample_bands():
    """Create sample Sentinel-2 band data."""
    import numpy as np

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


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
