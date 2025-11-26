"""
Coordinate Handler - GPS and Projection Utilities

Handles coordinate transformations and bounding box operations
for satellite data acquisition.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class BoundingBox:
    """Geographic bounding box."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point (lon, lat)."""
        return ((self.min_lon + self.max_lon) / 2, (self.min_lat + self.max_lat) / 2)

    @property
    def width(self) -> float:
        """Width in degrees."""
        return self.max_lon - self.min_lon

    @property
    def height(self) -> float:
        """Height in degrees."""
        return self.max_lat - self.min_lat

    def contains(self, lon: float, lat: float) -> bool:
        """Check if point is within bounding box."""
        return self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat

    def expand(self, factor: float) -> "BoundingBox":
        """Expand bounding box by factor."""
        center_lon, center_lat = self.center
        half_width = self.width / 2 * factor
        half_height = self.height / 2 * factor

        return BoundingBox(
            min_lon=center_lon - half_width,
            min_lat=center_lat - half_height,
            max_lon=center_lon + half_width,
            max_lat=center_lat + half_height,
        )

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to (min_lon, min_lat, max_lon, max_lat) tuple."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def to_polygon(self) -> List[Tuple[float, float]]:
        """Convert to polygon coordinates (for GeoJSON)."""
        return [
            (self.min_lon, self.min_lat),
            (self.max_lon, self.min_lat),
            (self.max_lon, self.max_lat),
            (self.min_lon, self.max_lat),
            (self.min_lon, self.min_lat),  # Close polygon
        ]


class CoordinateHandler:
    """
    Handler for coordinate transformations and operations.

    Supports WGS84 (EPSG:4326) and UTM projections.
    """

    # WGS84 ellipsoid parameters
    WGS84_A = 6378137.0  # Semi-major axis (meters)
    WGS84_F = 1 / 298.257223563  # Flattening
    WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis

    def __init__(self, default_epsg: int = 4326):
        """
        Initialize coordinate handler.

        Args:
            default_epsg: Default EPSG code (4326 for WGS84)
        """
        self.default_epsg = default_epsg

    def validate_coordinates(
        self,
        lon: float,
        lat: float,
    ) -> bool:
        """
        Validate WGS84 coordinates.

        Args:
            lon: Longitude (-180 to 180)
            lat: Latitude (-90 to 90)

        Returns:
            True if valid, False otherwise
        """
        return -180 <= lon <= 180 and -90 <= lat <= 90

    def degrees_to_meters(
        self,
        degrees: float,
        lat: float,
    ) -> float:
        """
        Convert degrees to approximate meters at given latitude.

        Args:
            degrees: Distance in degrees
            lat: Latitude for calculation

        Returns:
            Approximate distance in meters
        """
        # Approximate meters per degree at equator
        meters_per_degree = 111319.9

        # Adjust for latitude
        lat_rad = np.radians(lat)
        adjusted = meters_per_degree * np.cos(lat_rad)

        return degrees * adjusted

    def meters_to_degrees(
        self,
        meters: float,
        lat: float,
    ) -> float:
        """
        Convert meters to approximate degrees at given latitude.

        Args:
            meters: Distance in meters
            lat: Latitude for calculation

        Returns:
            Approximate distance in degrees
        """
        meters_per_degree = 111319.9
        lat_rad = np.radians(lat)
        adjusted = meters_per_degree * np.cos(lat_rad)

        if adjusted == 0:
            return 0

        return meters / adjusted

    def create_bounding_box(
        self,
        center_lon: float,
        center_lat: float,
        width_meters: float,
        height_meters: Optional[float] = None,
    ) -> BoundingBox:
        """
        Create a bounding box around a center point.

        Args:
            center_lon: Center longitude
            center_lat: Center latitude
            width_meters: Width in meters
            height_meters: Height in meters (defaults to width)

        Returns:
            BoundingBox object
        """
        if not self.validate_coordinates(center_lon, center_lat):
            raise ValueError(f"Invalid coordinates: ({center_lon}, {center_lat})")

        height_meters = height_meters or width_meters

        # Convert meters to degrees
        half_width = self.meters_to_degrees(width_meters / 2, center_lat)
        half_height = height_meters / 2 / 111319.9  # Latitude doesn't vary

        return BoundingBox(
            min_lon=center_lon - half_width,
            min_lat=center_lat - half_height,
            max_lon=center_lon + half_width,
            max_lat=center_lat + half_height,
        )

    def haversine_distance(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> float:
        """
        Calculate haversine distance between two points.

        Args:
            lon1, lat1: First point
            lon2, lat2: Second point

        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def get_utm_zone(self, lon: float) -> int:
        """
        Get UTM zone for a longitude.

        Args:
            lon: Longitude in degrees

        Returns:
            UTM zone number (1-60)
        """
        return int((lon + 180) / 6) + 1

    def get_utm_epsg(self, lon: float, lat: float) -> int:
        """
        Get UTM EPSG code for a location.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            EPSG code for appropriate UTM zone
        """
        zone = self.get_utm_zone(lon)

        # Northern hemisphere: 326XX, Southern: 327XX
        if lat >= 0:
            return 32600 + zone
        else:
            return 32700 + zone

    def transform_coordinates(
        self,
        lon: float,
        lat: float,
        from_epsg: int = 4326,
        to_epsg: int = 4326,
    ) -> Tuple[float, float]:
        """
        Transform coordinates between projections.

        Args:
            lon: X coordinate (longitude in WGS84)
            lat: Y coordinate (latitude in WGS84)
            from_epsg: Source EPSG code
            to_epsg: Target EPSG code

        Returns:
            Transformed (x, y) coordinates
        """
        try:
            from pyproj import Transformer

            transformer = Transformer.from_crs(
                f"EPSG:{from_epsg}",
                f"EPSG:{to_epsg}",
                always_xy=True,
            )

            x, y = transformer.transform(lon, lat)
            return (x, y)

        except ImportError:
            # Fallback: no transformation
            if from_epsg == to_epsg:
                return (lon, lat)
            raise ImportError("pyproj required for coordinate transformation")

    def parse_coordinate_string(
        self,
        coord_string: str,
    ) -> Tuple[float, float]:
        """
        Parse coordinate string to (lon, lat).

        Supports formats:
        - "lat, lon" or "lon, lat"
        - "12.345N, 67.890W"
        - "12°20'30\"N, 67°53'24\"W"

        Args:
            coord_string: Coordinate string

        Returns:
            (longitude, latitude) tuple
        """
        import re

        coord_string = coord_string.strip()

        # Try simple comma-separated
        if re.match(r"^-?\d+\.?\d*,\s*-?\d+\.?\d*$", coord_string):
            parts = coord_string.split(",")
            a, b = float(parts[0].strip()), float(parts[1].strip())

            # Assume lat, lon order if first is in lat range
            if -90 <= a <= 90:
                return (b, a)  # (lon, lat)
            return (a, b)

        # Try DMS format
        dms_pattern = r"(\d+)[°](\d+)['](\d+(?:\.\d+)?)[\"]\s*([NSEW])"
        matches = re.findall(dms_pattern, coord_string, re.IGNORECASE)

        if len(matches) == 2:
            coords = []
            for d, m, s, direction in matches:
                decimal = float(d) + float(m) / 60 + float(s) / 3600
                if direction.upper() in ["S", "W"]:
                    decimal = -decimal
                coords.append((decimal, direction.upper()))

            # Determine which is lat/lon
            lat = next(c[0] for c in coords if c[1] in ["N", "S"])
            lon = next(c[0] for c in coords if c[1] in ["E", "W"])
            return (lon, lat)

        raise ValueError(f"Could not parse coordinate string: {coord_string}")

    def format_coordinates(
        self,
        lon: float,
        lat: float,
        format: str = "decimal",
    ) -> str:
        """
        Format coordinates as string.

        Args:
            lon: Longitude
            lat: Latitude
            format: "decimal", "dms", or "dm"

        Returns:
            Formatted coordinate string
        """
        if format == "decimal":
            return f"{lat:.6f}, {lon:.6f}"

        elif format == "dms":

            def to_dms(val, directions):
                direction = directions[0] if val >= 0 else directions[1]
                val = abs(val)
                d = int(val)
                m = int((val - d) * 60)
                s = (val - d - m / 60) * 3600
                return f"{d}°{m}'{s:.2f}\"{direction}"

            lat_str = to_dms(lat, ("N", "S"))
            lon_str = to_dms(lon, ("E", "W"))
            return f"{lat_str}, {lon_str}"

        elif format == "dm":

            def to_dm(val, directions):
                direction = directions[0] if val >= 0 else directions[1]
                val = abs(val)
                d = int(val)
                m = (val - d) * 60
                return f"{d}°{m:.4f}'{direction}"

            lat_str = to_dm(lat, ("N", "S"))
            lon_str = to_dm(lon, ("E", "W"))
            return f"{lat_str}, {lon_str}"

        else:
            raise ValueError(f"Unknown format: {format}")
