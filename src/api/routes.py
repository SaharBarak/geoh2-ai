"""
API Routes

REST endpoints for H2 seep detection.
"""

import io
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    class_name: str
    class_id: int
    confidence: float
    probabilities: dict
    is_scd: bool
    image_path: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total: int
    scd_count: int
    processing_time_ms: float


class SpectralIndexRequest(BaseModel):
    """Request model for spectral index calculation."""
    indices: List[str] = Field(default=["ndvi", "bi"])


class SpectralIndexResponse(BaseModel):
    """Response model for spectral indices."""
    indices: dict
    metadata: dict


class CoordinateRequest(BaseModel):
    """Request model with coordinates."""
    longitude: float = Field(..., ge=-180, le=180)
    latitude: float = Field(..., ge=-90, le=90)
    width_meters: float = Field(default=640, ge=100, le=10000)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns API status and model information.
    """
    from .app import get_model

    model = get_model()

    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_name=model.config.name if model else None,
        device=model.device if model else None,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(..., description="Image file to classify"),
    confidence_threshold: float = Query(0.5, ge=0, le=1),
):
    """
    Classify a single image.

    Upload an image of a sub-circular depression to classify it
    as a potential H2 seep (SCD) or one of 8 other categories.

    - **file**: Image file (JPEG, PNG)
    - **confidence_threshold**: Minimum confidence threshold

    Returns classification result with probabilities for all 9 classes.
    """
    from .app import get_model

    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image
        contents = await file.read()

        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Run prediction
        result = model.predict(tmp_path)

        # Cleanup
        Path(tmp_path).unlink()

        return PredictionResponse(
            class_name=result.class_name,
            class_id=result.class_id,
            confidence=result.confidence,
            probabilities=result.probabilities,
            is_scd=result.is_scd(confidence_threshold),
            image_path=file.filename,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(..., description="Image files to classify"),
    confidence_threshold: float = Query(0.5, ge=0, le=1),
):
    """
    Classify multiple images in batch.

    Upload multiple images for efficient batch processing.

    - **files**: List of image files
    - **confidence_threshold**: Minimum confidence threshold

    Returns list of predictions with summary statistics.
    """
    import time
    from .app import get_model

    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch")

    start_time = time.time()

    try:
        # Save all files to temp directory
        tmp_paths = []
        for file in files:
            if not file.content_type.startswith("image/"):
                continue

            contents = await file.read()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(contents)
                tmp_paths.append((tmp.name, file.filename))

        # Run batch prediction
        paths = [p[0] for p in tmp_paths]
        results = model.predict_batch(paths)

        # Cleanup
        for path, _ in tmp_paths:
            Path(path).unlink()

        # Build response
        predictions = []
        scd_count = 0

        for result, (_, filename) in zip(results, tmp_paths):
            is_scd = result.is_scd(confidence_threshold)
            if is_scd:
                scd_count += 1

            predictions.append(PredictionResponse(
                class_name=result.class_name,
                class_id=result.class_id,
                confidence=result.confidence,
                probabilities=result.probabilities,
                is_scd=is_scd,
                image_path=filename,
            ))

        elapsed_ms = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            scd_count=scd_count,
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/coordinates")
async def predict_from_coordinates(
    request: CoordinateRequest,
    source: str = Query("google_maps", enum=["google_maps", "sentinel2"]),
):
    """
    Fetch imagery and classify for given coordinates.

    Downloads satellite imagery for the specified location
    and runs classification.

    - **longitude**: Longitude (-180 to 180)
    - **latitude**: Latitude (-90 to 90)
    - **width_meters**: Area width in meters
    - **source**: Imagery source (google_maps or sentinel2)
    """
    from .app import get_model

    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if source == "google_maps":
            from src.data import GoogleMapsScraper

            scraper = GoogleMapsScraper()
            image_result = scraper.download(
                request.longitude,
                request.latitude,
            )

            if image_result is None:
                raise HTTPException(status_code=404, detail="Could not fetch imagery")

            image = image_result.image

        else:  # sentinel2
            from src.data import Sentinel2Fetcher

            fetcher = Sentinel2Fetcher()
            image_result = fetcher.fetch(
                request.longitude,
                request.latitude,
                width_meters=request.width_meters,
            )

            if image_result is None:
                raise HTTPException(status_code=404, detail="Could not fetch imagery")

            image = image_result.get_rgb()

        # Run prediction
        result = model.predict(image)

        return {
            "prediction": PredictionResponse(
                class_name=result.class_name,
                class_id=result.class_id,
                confidence=result.confidence,
                probabilities=result.probabilities,
                is_scd=result.is_scd(),
            ),
            "location": {
                "longitude": request.longitude,
                "latitude": request.latitude,
            },
            "source": source,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spectral-indices", response_model=SpectralIndexResponse)
async def calculate_spectral_indices(
    file: UploadFile = File(..., description="Sentinel-2 image file"),
    indices: str = Query("ndvi,bi", description="Comma-separated list of indices"),
):
    """
    Calculate spectral indices from Sentinel-2 data.

    Compute vegetation and brightness indices from multispectral
    satellite imagery.

    - **file**: Sentinel-2 GeoTIFF file
    - **indices**: Indices to compute (ndvi, bi, ndwi, savi, etc.)
    """
    try:
        from src.preprocessing import SpectralIndexCalculator
        import rasterio

        # Parse requested indices
        index_list = [i.strip().lower() for i in indices.split(",")]

        # Read file
        contents = await file.read()

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Load bands
        with rasterio.open(tmp_path) as src:
            # Assume standard band order or use band descriptions
            bands = {}
            band_mapping = {1: "B2", 2: "B3", 3: "B4", 4: "B8"}

            for i in range(1, min(src.count + 1, 5)):
                band_name = band_mapping.get(i, f"B{i}")
                bands[band_name] = src.read(i).astype(np.float32)

        # Cleanup
        Path(tmp_path).unlink()

        # Calculate indices
        calculator = SpectralIndexCalculator()
        results = calculator.compute_multiple(index_list, bands)

        # Convert results to serializable format
        index_values = {}
        for name, result in results.items():
            index_values[name] = {
                "mean": float(np.nanmean(result.value)),
                "std": float(np.nanstd(result.value)),
                "min": float(np.nanmin(result.value)),
                "max": float(np.nanmax(result.value)),
                "valid_range": result.valid_range,
            }

        return SpectralIndexResponse(
            indices=index_values,
            metadata={
                "computed_indices": list(results.keys()),
                "available_bands": list(bands.keys()),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model.

    Returns model architecture, parameters, and configuration.
    """
    from .app import get_model

    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return model.get_model_info()


@router.get("/classes")
async def get_class_names():
    """
    Get list of classification classes.

    Returns the 9 classes used for classification:
    - SCD (H2-related sub-circular depression)
    - 8 similar-looking but non-H2 structures
    """
    from src.models import DEFAULT_CLASS_NAMES

    return {
        "classes": [
            {"id": i, "name": name, "is_positive": name == "SCD"}
            for i, name in enumerate(DEFAULT_CLASS_NAMES)
        ],
        "total": len(DEFAULT_CLASS_NAMES),
    }
