"""
Integration Tests - API Endpoints

Tests the FastAPI REST endpoints.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.api import create_app

        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image as bytes."""
        import cv2

        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Encode as JPEG
        _, buffer = cv2.imencode(".jpg", img)
        return buffer.tobytes()

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

    def test_get_classes(self, client):
        """Test getting class names."""
        response = client.get("/api/v1/classes")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 9
        assert len(data["classes"]) == 9

        # Check SCD is marked as positive
        scd_class = next(c for c in data["classes"] if c["name"] == "SCD")
        assert scd_class["is_positive"] is True

    def test_predict_endpoint_validation(self, client):
        """Test prediction endpoint input validation."""
        # Test with no file
        response = client.post("/api/v1/predict")
        assert response.status_code == 422  # Validation error

        # Test with non-image file
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        # 400 if validation fails, 503 if model not loaded
        assert response.status_code in [400, 503]

    def test_predict_single_image(self, client, sample_image_bytes):
        """Test single image prediction."""
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

        # May return 503 if model not loaded, 200 if successful
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "class_name" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "is_scd" in data
            assert 0 <= data["confidence"] <= 1

    def test_predict_with_threshold(self, client, sample_image_bytes):
        """Test prediction with custom threshold."""
        response = client.post(
            "/api/v1/predict?confidence_threshold=0.7",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

        assert response.status_code in [200, 503]

    def test_batch_predict(self, client, sample_image_bytes):
        """Test batch prediction endpoint."""
        files = [
            ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("test2.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("test3.jpg", sample_image_bytes, "image/jpeg")),
        ]

        response = client.post("/api/v1/predict/batch", files=files)

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total" in data
            assert "scd_count" in data
            assert data["total"] == 3

    def test_batch_limit(self, client, sample_image_bytes):
        """Test batch size limit."""
        # Create 101 files (over limit)
        files = [
            ("files", (f"test{i}.jpg", sample_image_bytes, "image/jpeg"))
            for i in range(101)
        ]

        response = client.post("/api/v1/predict/batch", files=files)

        # Should reject as too many files
        assert response.status_code in [400, 503]

    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")

        # May return 503 if model not loaded
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "name" in data
            assert "architecture" in data


class TestAPIErrorHandling:
    """Test API error handling."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.api import create_app

        app = create_app()
        return TestClient(app)

    def test_invalid_endpoint(self, client):
        """Test 404 for invalid endpoint."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_invalid_method(self, client):
        """Test 405 for invalid method."""
        response = client.delete("/api/v1/health")
        assert response.status_code == 405

    def test_malformed_image(self, client):
        """Test handling of malformed image."""
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test.jpg", b"not valid jpeg data", "image/jpeg")},
        )

        # Should handle gracefully
        assert response.status_code in [400, 500, 503]
