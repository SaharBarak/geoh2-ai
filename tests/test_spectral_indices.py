"""
Tests for Spectral Indices Calculator
"""

import numpy as np
import pytest

from src.preprocessing.spectral_indices import SpectralIndexCalculator


class TestSpectralIndexCalculator:
    """Test suite for SpectralIndexCalculator"""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance"""
        return SpectralIndexCalculator()

    @pytest.fixture
    def sample_bands(self):
        """Create sample spectral bands"""
        height, width = 100, 100
        return {
            "B2": np.random.rand(height, width) * 0.3,  # Blue
            "B3": np.random.rand(height, width) * 0.4,  # Green
            "B4": np.random.rand(height, width) * 0.5,  # Red
            "B8": np.random.rand(height, width) * 0.6,  # NIR
            "B11": np.random.rand(height, width) * 0.5,  # SWIR1
            "B12": np.random.rand(height, width) * 0.4,  # SWIR2
        }

    def test_ndvi_computation(self, calculator, sample_bands):
        """Test NDVI calculation"""
        result = calculator.ndvi(sample_bands["B8"], sample_bands["B4"])

        assert result.name == "NDVI"
        assert result.value.shape == (100, 100)
        assert result.valid_range == (-1.0, 1.0)
        assert np.all(result.value >= -1.0) and np.all(result.value <= 1.0)

    def test_brightness_index(self, calculator, sample_bands):
        """Test Brightness Index calculation"""
        result = calculator.brightness_index(sample_bands["B4"], sample_bands["B8"])

        assert result.name == "BI"
        assert result.value.shape == (100, 100)
        assert np.all(result.value >= 0.0)

    def test_safe_divide(self, calculator):
        """Test safe division handling"""
        numerator = np.array([1.0, 2.0, 3.0])
        denominator = np.array([2.0, 0.0, 4.0])

        result = calculator._safe_divide(numerator, denominator, fill_value=0.0)

        assert result[0] == 0.5
        assert result[1] == 0.0  # Division by zero
        assert result[2] == 0.75

    def test_compute_method(self, calculator, sample_bands):
        """Test generic compute method"""
        result = calculator.compute("ndvi", sample_bands)

        assert isinstance(result.value, np.ndarray)
        assert result.name == "NDVI"

    def test_compute_multiple(self, calculator, sample_bands):
        """Test computing multiple indices"""
        indices = ["ndvi", "bi", "ndwi"]
        results = calculator.compute_multiple(indices, sample_bands)

        assert len(results) == 3
        assert all(name in results for name in indices)

    def test_compute_multiple_stacked(self, calculator, sample_bands):
        """Test stacked output"""
        indices = ["ndvi", "bi"]
        stacked = calculator.compute_multiple(indices, sample_bands, stack=True)

        assert stacked.shape == (100, 100, 2)

    def test_normalize(self, calculator, sample_bands):
        """Test normalization"""
        result = calculator.ndvi(sample_bands["B8"], sample_bands["B4"])
        normalized = result.normalize()

        assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)

    def test_to_uint8(self, calculator, sample_bands):
        """Test conversion to 8-bit"""
        result = calculator.ndvi(sample_bands["B8"], sample_bands["B4"])
        uint8_image = result.to_uint8()

        assert uint8_image.dtype == np.uint8
        assert np.all(uint8_image >= 0) and np.all(uint8_image <= 255)

    def test_invalid_index_name(self, calculator, sample_bands):
        """Test error handling for invalid index"""
        with pytest.raises(ValueError):
            calculator.compute("invalid_index", sample_bands)

    def test_available_indices(self, calculator):
        """Test getting available indices"""
        indices = calculator.available_indices

        assert "ndvi" in indices
        assert "bi" in indices
        assert len(indices) >= 9  # Should have at least 9 indices
