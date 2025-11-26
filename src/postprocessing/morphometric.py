"""
Morphometric Analysis for Sub-Circular Depressions
Analyzes shape, size, and depth characteristics
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class MorphometricFeatures:
    """Morphometric features of a structure"""

    area: float  # in square meters
    perimeter: float  # in meters
    diameter: float  # in meters
    circularity: float  # 0-1, 1 is perfect circle
    elongation: float  # aspect ratio
    depth: Optional[float] = None  # in meters (if DEM available)

    def is_valid_scd(
        self,
        min_diameter: float = 50.0,
        max_diameter: float = 1000.0,
        min_circularity: float = 0.6,
    ) -> bool:
        """
        Check if morphometry matches SCD criteria.

        Args:
            min_diameter: Minimum diameter in meters
            max_diameter: Maximum diameter in meters
            min_circularity: Minimum circularity score

        Returns:
            True if features match SCD criteria
        """
        size_valid = min_diameter <= self.diameter <= max_diameter
        shape_valid = self.circularity >= min_circularity

        return size_valid and shape_valid

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "area": float(self.area),
            "perimeter": float(self.perimeter),
            "diameter": float(self.diameter),
            "circularity": float(self.circularity),
            "elongation": float(self.elongation),
            "depth": float(self.depth) if self.depth else None,
        }


class MorphometricAnalyzer:
    """
    Analyzes morphometric properties of structures.
    Helps filter false positives based on shape/size.
    """

    def __init__(
        self,
        pixel_resolution: float = 10.0,  # meters per pixel
        min_diameter_m: float = 50.0,
        max_diameter_m: float = 1000.0,
        min_circularity: float = 0.6,
    ):
        """
        Initialize analyzer.

        Args:
            pixel_resolution: Spatial resolution in meters/pixel
            min_diameter_m: Minimum valid diameter in meters
            max_diameter_m: Maximum valid diameter in meters
            min_circularity: Minimum circularity threshold
        """
        self.pixel_resolution = pixel_resolution
        self.min_diameter_m = min_diameter_m
        self.max_diameter_m = max_diameter_m
        self.min_circularity = min_circularity

    def analyze_binary_mask(
        self, mask: np.ndarray, dem: Optional[np.ndarray] = None
    ) -> MorphometricFeatures:
        """
        Analyze morphometric features from binary mask.

        Args:
            mask: Binary mask of structure (1=structure, 0=background)
            dem: Optional digital elevation model

        Returns:
            MorphometricFeatures object
        """
        # Compute area
        area_pixels = np.sum(mask)
        area_m2 = area_pixels * (self.pixel_resolution**2)

        # Compute perimeter using edge detection
        from scipy import ndimage

        eroded = ndimage.binary_erosion(mask)
        boundary = mask.astype(int) - eroded.astype(int)
        perimeter_pixels = np.sum(boundary)
        perimeter_m = perimeter_pixels * self.pixel_resolution

        # Compute equivalent diameter
        diameter_m = 2 * np.sqrt(area_m2 / np.pi)

        # Compute circularity (4π * area / perimeter^2)
        if perimeter_m > 0:
            circularity = (4 * np.pi * area_m2) / (perimeter_m**2)
            circularity = min(circularity, 1.0)  # Clamp to [0, 1]
        else:
            circularity = 0.0

        # Compute elongation (aspect ratio)
        labeled, _ = ndimage.label(mask)
        if labeled.max() > 0:
            # Get bounding box
            slices = ndimage.find_objects(labeled)[0]
            height = slices[0].stop - slices[0].start
            width = slices[1].stop - slices[1].start
            elongation = max(height, width) / max(min(height, width), 1)
        else:
            elongation = 1.0

        # Compute depth if DEM provided
        depth = None
        if dem is not None:
            depth = self._estimate_depth(mask, dem)

        return MorphometricFeatures(
            area=area_m2,
            perimeter=perimeter_m,
            diameter=diameter_m,
            circularity=circularity,
            elongation=elongation,
            depth=depth,
        )

    def _estimate_depth(self, mask: np.ndarray, dem: np.ndarray) -> float:
        """
        Estimate depression depth from DEM.

        Args:
            mask: Binary mask of structure
            dem: Digital elevation model

        Returns:
            Estimated depth in meters
        """
        # Get elevations inside and outside depression
        inside = dem[mask > 0]

        # Get rim elevations (dilate mask and subtract)
        from scipy.ndimage import binary_dilation

        dilated = binary_dilation(mask, iterations=3)
        rim_mask = dilated & ~mask
        outside = dem[rim_mask]

        if len(inside) == 0 or len(outside) == 0:
            return 0.0

        # Depth is difference between rim and floor
        rim_elevation = np.median(outside)
        floor_elevation = np.median(inside)
        depth = max(rim_elevation - floor_elevation, 0.0)

        return depth

    def filter_predictions(self, predictions: list, masks: list[np.ndarray]) -> list:
        """
        Filter predictions based on morphometric criteria.

        Args:
            predictions: List of PredictionResult objects
            masks: List of binary masks for each prediction

        Returns:
            Filtered list of predictions
        """
        filtered = []

        for pred, mask in zip(predictions, masks):
            features = self.analyze_binary_mask(mask)

            # Check if morphometry is valid
            if features.is_valid_scd(
                min_diameter=self.min_diameter_m,
                max_diameter=self.max_diameter_m,
                min_circularity=self.min_circularity,
            ):
                # Add morphometric metadata
                if pred.metadata is None:
                    pred.metadata = {}
                pred.metadata["morphometry"] = features.to_dict()
                filtered.append(pred)

        return filtered
