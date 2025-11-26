"""
Geological Filter - Context-Based Refinement

Refines predictions using geological context:
- Sedimentary basin overlay
- Fault zone proximity
- Rock type correlation
- Elevation/terrain analysis
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class GeologicalContext:
    """Geological context for a location."""

    in_sedimentary_basin: bool
    basin_name: Optional[str]
    distance_to_fault_km: float
    fault_name: Optional[str]
    rock_type: Optional[str]
    elevation_m: float
    terrain_slope: float
    geological_age: Optional[str]

    def is_favorable_for_h2(self) -> bool:
        """Check if geological context is favorable for H2 seeps."""
        # Based on research: H2 seeps associated with sedimentary basins
        # and proximity to faults
        return (
            self.in_sedimentary_basin
            and self.distance_to_fault_km < 50  # Within 50km of fault
        )


@dataclass
class FilterResult:
    """Result from geological filtering."""

    passed: bool
    confidence_adjustment: float
    reasons: List[str]
    geological_context: Optional[GeologicalContext]


class GeologicalFilter:
    """
    Geological context filter for H2 seep predictions.

    Refines predictions based on geological favorability:
    - Increases confidence for structures in sedimentary basins
    - Decreases confidence for geologically unfavorable areas
    - Flags structures near known fault zones
    """

    def __init__(
        self,
        basins_shapefile: Optional[str] = None,
        faults_shapefile: Optional[str] = None,
        dem_path: Optional[str] = None,
    ):
        """
        Initialize geological filter.

        Args:
            basins_shapefile: Path to sedimentary basins shapefile
            faults_shapefile: Path to fault lines shapefile
            dem_path: Path to digital elevation model
        """
        self.basins_shapefile = basins_shapefile
        self.faults_shapefile = faults_shapefile
        self.dem_path = dem_path

        self._basins = None
        self._faults = None
        self._dem = None

        # Load data if paths provided
        self._load_data()

    def _load_data(self) -> None:
        """Load geological data layers."""
        try:
            import geopandas as gpd

            if self.basins_shapefile and Path(self.basins_shapefile).exists():
                self._basins = gpd.read_file(self.basins_shapefile)

            if self.faults_shapefile and Path(self.faults_shapefile).exists():
                self._faults = gpd.read_file(self.faults_shapefile)

        except ImportError:
            print("GeoPandas not available. Geological filtering limited.")

        try:
            import rasterio

            if self.dem_path and Path(self.dem_path).exists():
                self._dem = rasterio.open(self.dem_path)

        except ImportError:
            pass

    def filter(
        self,
        prediction,
        lon: float,
        lat: float,
    ) -> FilterResult:
        """
        Apply geological filter to a prediction.

        Args:
            prediction: PredictionResult object
            lon: Longitude of structure
            lat: Latitude of structure

        Returns:
            FilterResult with filtering decision
        """
        reasons = []
        confidence_adjustment = 0.0

        # Get geological context
        context = self.get_geological_context(lon, lat)

        if context is None:
            # No geological data available
            return FilterResult(
                passed=True,
                confidence_adjustment=0.0,
                reasons=["No geological data available"],
                geological_context=None,
            )

        # Check sedimentary basin
        if context.in_sedimentary_basin:
            confidence_adjustment += 0.1
            reasons.append(f"In sedimentary basin: {context.basin_name or 'Unknown'}")
        else:
            confidence_adjustment -= 0.15
            reasons.append("Not in sedimentary basin")

        # Check fault proximity
        if context.distance_to_fault_km < 10:
            confidence_adjustment += 0.1
            reasons.append(f"Near fault ({context.distance_to_fault_km:.1f} km)")
        elif context.distance_to_fault_km < 50:
            confidence_adjustment += 0.05
            reasons.append(f"Within 50km of fault")
        else:
            confidence_adjustment -= 0.05
            reasons.append(
                f"Far from known faults ({context.distance_to_fault_km:.1f} km)"
            )

        # Check rock type (if available)
        favorable_rocks = ["sedimentary", "sandstone", "limestone", "shale"]
        if context.rock_type:
            if any(r in context.rock_type.lower() for r in favorable_rocks):
                confidence_adjustment += 0.05
                reasons.append(f"Favorable rock type: {context.rock_type}")
            else:
                confidence_adjustment -= 0.05
                reasons.append(f"Unfavorable rock type: {context.rock_type}")

        # Check terrain
        if context.terrain_slope > 30:
            confidence_adjustment -= 0.1
            reasons.append(f"Steep terrain ({context.terrain_slope:.1f}°)")

        # Determine if passed
        # Apply adjustment to original confidence
        original_confidence = (
            prediction.confidence if hasattr(prediction, "confidence") else 0.5
        )
        adjusted_confidence = original_confidence + confidence_adjustment

        # Pass if adjusted confidence still above threshold
        passed = adjusted_confidence >= 0.5 and context.is_favorable_for_h2()

        return FilterResult(
            passed=passed,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons,
            geological_context=context,
        )

    def get_geological_context(
        self,
        lon: float,
        lat: float,
    ) -> Optional[GeologicalContext]:
        """
        Get geological context for a location.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            GeologicalContext or None
        """
        try:
            from shapely.geometry import Point

            point = Point(lon, lat)

            # Check basin
            in_basin = False
            basin_name = None

            if self._basins is not None:
                for _, basin in self._basins.iterrows():
                    if basin.geometry.contains(point):
                        in_basin = True
                        basin_name = basin.get("name", basin.get("NAME", "Unknown"))
                        break

            # Check fault distance
            fault_distance = float("inf")
            fault_name = None

            if self._faults is not None:
                for _, fault in self._faults.iterrows():
                    dist = point.distance(fault.geometry)
                    # Convert degrees to km (approximate)
                    dist_km = dist * 111
                    if dist_km < fault_distance:
                        fault_distance = dist_km
                        fault_name = fault.get("name", fault.get("NAME"))

            # Get elevation
            elevation = self._get_elevation(lon, lat)
            slope = self._get_slope(lon, lat)

            return GeologicalContext(
                in_sedimentary_basin=in_basin,
                basin_name=basin_name,
                distance_to_fault_km=fault_distance
                if fault_distance != float("inf")
                else 1000,
                fault_name=fault_name,
                rock_type=None,  # Would require geological map
                elevation_m=elevation,
                terrain_slope=slope,
                geological_age=None,
            )

        except Exception as e:
            print(f"Error getting geological context: {e}")
            return None

    def _get_elevation(self, lon: float, lat: float) -> float:
        """Get elevation at a point from DEM."""
        if self._dem is None:
            return 0.0

        try:
            row, col = self._dem.index(lon, lat)
            elevation = self._dem.read(1)[row, col]
            return float(elevation)
        except Exception:
            return 0.0

    def _get_slope(self, lon: float, lat: float) -> float:
        """Calculate slope at a point from DEM."""
        if self._dem is None:
            return 0.0

        try:
            # Get 3x3 window around point
            row, col = self._dem.index(lon, lat)
            data = self._dem.read(1)

            if (
                row > 0
                and row < data.shape[0] - 1
                and col > 0
                and col < data.shape[1] - 1
            ):
                window = data[row - 1 : row + 2, col - 1 : col + 2]
                dz_dx = (window[1, 2] - window[1, 0]) / (2 * self._dem.res[0])
                dz_dy = (window[2, 1] - window[0, 1]) / (2 * self._dem.res[1])
                slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
                return float(slope)

            return 0.0

        except Exception:
            return 0.0

    def filter_batch(
        self,
        predictions: List,
        coordinates: List[Tuple[float, float]],
    ) -> List[FilterResult]:
        """
        Apply geological filter to multiple predictions.

        Args:
            predictions: List of PredictionResult objects
            coordinates: List of (lon, lat) tuples

        Returns:
            List of FilterResult objects
        """
        results = []

        for pred, (lon, lat) in zip(predictions, coordinates):
            result = self.filter(pred, lon, lat)
            results.append(result)

        return results


class KnownH2FieldsChecker:
    """
    Check proximity to known H2 fields.

    Increases confidence for structures near known H2 occurrences.
    """

    # Known H2 seep locations from literature
    KNOWN_H2_LOCATIONS = [
        # Brazil - São Francisco Basin
        {"name": "São Francisco Basin", "lon": -44.5, "lat": -15.5, "radius_km": 200},
        # Russia
        {"name": "Western Siberia", "lon": 70.0, "lat": 60.0, "radius_km": 100},
        # USA
        {"name": "Kansas", "lon": -99.0, "lat": 38.0, "radius_km": 50},
        # Namibia - Fairy circles region
        {"name": "Namibia", "lon": 15.5, "lat": -24.5, "radius_km": 100},
        # Mali - Bourakébougou
        {"name": "Bourakébougou", "lon": -5.5, "lat": 13.5, "radius_km": 20},
    ]

    def __init__(self, custom_locations: Optional[List[Dict]] = None):
        """
        Initialize checker.

        Args:
            custom_locations: Additional known H2 locations
        """
        self.locations = self.KNOWN_H2_LOCATIONS.copy()
        if custom_locations:
            self.locations.extend(custom_locations)

    def check(self, lon: float, lat: float) -> Tuple[bool, Optional[str], float]:
        """
        Check if location is near known H2 field.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            Tuple of (is_near, field_name, distance_km)
        """
        from src.preprocessing.coordinate_handler import CoordinateHandler

        handler = CoordinateHandler()

        for location in self.locations:
            distance = (
                handler.haversine_distance(lon, lat, location["lon"], location["lat"])
                / 1000
            )  # Convert to km

            if distance <= location["radius_km"]:
                return True, location["name"], distance

        return False, None, float("inf")
