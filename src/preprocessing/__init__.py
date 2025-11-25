"""
H2 Seep Detection - Preprocessing Module

Provides spectral index calculations and image preprocessing utilities.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np


class SpectralBand(Enum):
    """Sentinel-2 spectral bands."""
    B1 = "B1"    # Coastal aerosol (60m)
    B2 = "B2"    # Blue (10m)
    B3 = "B3"    # Green (10m)
    B4 = "B4"    # Red (10m)
    B5 = "B5"    # Vegetation Red Edge (20m)
    B6 = "B6"    # Vegetation Red Edge (20m)
    B7 = "B7"    # Vegetation Red Edge (20m)
    B8 = "B8"    # NIR (10m)
    B8A = "B8A"  # Vegetation Red Edge (20m)
    B9 = "B9"    # Water Vapour (60m)
    B10 = "B10"  # SWIR - Cirrus (60m)
    B11 = "B11"  # SWIR (20m)
    B12 = "B12"  # SWIR (20m)


@dataclass
class IndexResult:
    """Result of spectral index computation."""
    name: str
    value: np.ndarray
    valid_range: Tuple[float, float]
    metadata: Optional[Dict] = None

    def to_uint8(self) -> np.ndarray:
        """Convert to 8-bit image for visualization."""
        min_val, max_val = self.valid_range
        normalized = (self.value - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)
        return (normalized * 255).astype(np.uint8)

    def mask_invalid(self, fill_value: float = np.nan) -> np.ndarray:
        """Mask values outside valid range."""
        min_val, max_val = self.valid_range
        result = self.value.copy()
        result[(result < min_val) | (result > max_val)] = fill_value
        return result


# Index definitions with formulas and descriptions
INDEX_DEFINITIONS = {
    "ndvi": {
        "name": "Normalized Difference Vegetation Index",
        "formula": "(NIR - RED) / (NIR + RED)",
        "bands": ["B8", "B4"],
        "valid_range": (-1, 1),
        "description": "Primary index for vegetation anomaly detection (H2 seeps)",
    },
    "bi": {
        "name": "Brightness Index",
        "formula": "sqrt((RED^2 + NIR^2) / 2)",
        "bands": ["B4", "B8"],
        "valid_range": (0, 1),
        "description": "Secondary index for brightness/salt distinction",
    },
    "ndwi": {
        "name": "Normalized Difference Water Index",
        "formula": "(GREEN - NIR) / (GREEN + NIR)",
        "bands": ["B3", "B8"],
        "valid_range": (-1, 1),
        "description": "Water detection",
    },
    "savi": {
        "name": "Soil Adjusted Vegetation Index",
        "formula": "((NIR - RED) / (NIR + RED + L)) * (1 + L), L=0.5",
        "bands": ["B8", "B4"],
        "valid_range": (-1, 1),
        "description": "Reduced soil influence on vegetation signal",
    },
    "evi": {
        "name": "Enhanced Vegetation Index",
        "formula": "2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)",
        "bands": ["B8", "B4", "B2"],
        "valid_range": (-1, 1),
        "description": "Enhanced vegetation detection",
    },
    "ndmi": {
        "name": "Normalized Difference Moisture Index",
        "formula": "(NIR - SWIR) / (NIR + SWIR)",
        "bands": ["B8", "B11"],
        "valid_range": (-1, 1),
        "description": "Vegetation water content",
    },
    "si": {
        "name": "Salinity Index",
        "formula": "sqrt(GREEN * RED)",
        "bands": ["B3", "B4"],
        "valid_range": (0, 1),
        "description": "Salinity detection",
    },
    "ndsi": {
        "name": "Normalized Difference Salinity Index",
        "formula": "(RED - NIR) / (RED + NIR)",
        "bands": ["B4", "B8"],
        "valid_range": (-1, 1),
        "description": "Normalized salinity measure",
    },
    "nbr": {
        "name": "Normalized Burn Ratio",
        "formula": "(NIR - SWIR2) / (NIR + SWIR2)",
        "bands": ["B8", "B12"],
        "valid_range": (-1, 1),
        "description": "Burn severity and vegetation dryness",
    },
}


from .spectral_indices import SpectralIndexCalculator
from .image_processor import ImageProcessor
from .coordinate_handler import CoordinateHandler

__all__ = [
    "SpectralBand",
    "IndexResult",
    "INDEX_DEFINITIONS",
    "SpectralIndexCalculator",
    "ImageProcessor",
    "CoordinateHandler",
]
