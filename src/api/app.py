"""
FastAPI Application

Main API application with endpoints for H2 seep detection.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


# Global model instance
_model = None


def get_model():
    """Get the global model instance."""
    global _model
    return _model


def set_model(model):
    """Set the global model instance."""
    global _model
    _model = model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Loads model on startup, cleans up on shutdown.
    """
    # Startup
    print("Loading H2 seep detection model...")

    from src.models import ModelFactory, ModelConfig

    # Load model configuration
    config_path = os.environ.get("MODEL_CONFIG", "config/model_config.yaml")
    weights_path = os.environ.get("MODEL_WEIGHTS", None)

    try:
        if Path(config_path).exists():
            model = ModelFactory.create_from_yaml(config_path)
        else:
            # Create default model
            model = ModelFactory.create_default(weights_path)

        set_model(model)
        print(f"Model loaded: {model.config.name}")
        print(f"Device: {model.device}")

    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will run without model - some endpoints will be unavailable")

    yield

    # Shutdown
    print("Shutting down H2 seep detection API...")
    set_model(None)


def create_app(
    title: str = "H2 Seep Detection API",
    version: str = "0.1.0",
    debug: bool = False,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        title: API title
        version: API version
        debug: Enable debug mode

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description="""
        API for detecting potential hydrogen seeps from satellite imagery.

        Based on Ginzburg et al. (2025) methodology achieving 90% accuracy
        on Google Maps imagery classification.

        ## Features
        - Single image prediction
        - Batch processing
        - Spectral index calculation
        - Geological context filtering
        """,
        version=version,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router, prefix="/api/v1")

    return app


# Default app instance
app = create_app()
