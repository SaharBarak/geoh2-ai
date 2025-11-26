"""
Tests for Model Components
"""

import numpy as np
import pytest
import torch

from src.models import ModelConfig, PredictionResult


class TestPredictionResult:
    """Test PredictionResult dataclass"""

    @pytest.fixture
    def sample_result(self):
        """Create sample prediction result"""
        return PredictionResult(
            class_name="SCD",
            class_id=0,
            confidence=0.85,
            probabilities={
                "SCD": 0.85,
                "salt_lake": 0.10,
                "impact_crater": 0.05,
            },
        )

    def test_is_scd_positive(self, sample_result):
        """Test SCD detection with high confidence"""
        assert sample_result.is_scd(threshold=0.5) is True
        assert sample_result.is_scd(threshold=0.9) is False

    def test_is_scd_negative(self):
        """Test non-SCD detection"""
        result = PredictionResult(
            class_name="salt_lake",
            class_id=7,
            confidence=0.90,
            probabilities={"salt_lake": 0.90, "SCD": 0.10},
        )

        assert result.is_scd(threshold=0.5) is False

    def test_to_dict(self, sample_result):
        """Test conversion to dictionary"""
        result_dict = sample_result.to_dict()

        assert result_dict["class_name"] == "SCD"
        assert result_dict["confidence"] == 0.85
        assert result_dict["is_scd"] is True
        assert "probabilities" in result_dict


class TestModelConfig:
    """Test ModelConfig dataclass"""

    @pytest.fixture
    def sample_config(self):
        """Create sample model configuration"""
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
        )

    def test_config_creation(self, sample_config):
        """Test basic configuration creation"""
        assert sample_config.name == "test_model"
        assert sample_config.num_classes == 9
        assert len(sample_config.class_names) == 9

    def test_positive_class_idx(self, sample_config):
        """Test positive class index"""
        assert sample_config.positive_class_idx == 0
        assert sample_config.class_names[sample_config.positive_class_idx] == "SCD"

    def test_device_default(self, sample_config):
        """Test default device selection"""
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert sample_config.device == expected_device
