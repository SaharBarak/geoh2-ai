"""
H2 Seep Detection API

FastAPI-based REST API for model inference and batch processing.
"""

from .app import create_app, app
from .routes import router

__all__ = ["create_app", "app", "router"]
