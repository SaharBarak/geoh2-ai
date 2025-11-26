"""
Spatial Statistics and Clustering Analysis
Analyzes spatial distribution of predicted H2 seeps
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


@dataclass
class Cluster:
    """Represents a spatial cluster of structures"""

    cluster_id: int
    size: int
    centroid: Tuple[float, float]  # (lon, lat)
    members: List[int]  # Indices of structures in cluster
    avg_confidence: float
    scd_percentage: float  # Percentage classified as SCD

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "centroid": self.centroid,
            "avg_confidence": float(self.avg_confidence),
            "scd_percentage": float(self.scd_percentage),
            "is_high_priority": self.is_high_priority(),
        }

    def is_high_priority(self, threshold: float = 0.7) -> bool:
        """Check if cluster is high priority for field validation"""
        return self.scd_percentage >= threshold and self.size >= 3


class SpatialAnalyzer:
    """
    Analyzes spatial patterns in predictions.
    Identifies clusters and spatial relationships.
    """

    def __init__(self, distance_threshold_km: float = 5.0, min_cluster_size: int = 3):
        """
        Initialize spatial analyzer.

        Args:
            distance_threshold_km: Max distance for clustering (km)
            min_cluster_size: Minimum structures per cluster
        """
        self.distance_threshold_km = distance_threshold_km
        self.min_cluster_size = min_cluster_size

    def haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate haversine distance between two coordinates.

        Args:
            coord1: (lon, lat) in degrees
            coord2: (lon, lat) in degrees

        Returns:
            Distance in kilometers
        """
        lon1, lat1 = np.radians(coord1)
        lon2, lat2 = np.radians(coord2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in km
        r = 6371.0

        return c * r

    def compute_distance_matrix(self, coordinates: List[Tuple[float, float]]) -> np.ndarray:
        """
        Compute pairwise distance matrix.

        Args:
            coordinates: List of (lon, lat) tuples

        Returns:
            Distance matrix in kilometers
        """
        n = len(coordinates)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.haversine_distance(coordinates[i], coordinates[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def cluster_structures(
        self, coordinates: List[Tuple[float, float]], predictions: Optional[List] = None
    ) -> Tuple[List[Cluster], np.ndarray]:
        """
        Cluster structures using DBSCAN.

        Args:
            coordinates: List of (lon, lat) coordinates
            predictions: Optional list of PredictionResult objects

        Returns:
            Tuple of (clusters list, labels array)
        """
        if len(coordinates) < self.min_cluster_size:
            return [], np.array([-1] * len(coordinates))

        # Convert to radians for clustering
        coords_rad = np.radians(coordinates)

        # DBSCAN clustering
        # eps in radians: distance_km / earth_radius_km
        eps_rad = self.distance_threshold_km / 6371.0

        clustering = DBSCAN(eps=eps_rad, min_samples=self.min_cluster_size, metric="haversine").fit(
            coords_rad
        )

        labels = clustering.labels_

        # Build cluster objects
        clusters = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        for label in unique_labels:
            mask = labels == label
            member_indices = np.where(mask)[0].tolist()

            # Compute centroid
            cluster_coords = np.array(coordinates)[mask]
            centroid = tuple(np.mean(cluster_coords, axis=0))

            # Compute statistics
            size = len(member_indices)

            if predictions:
                cluster_preds = [predictions[i] for i in member_indices]
                avg_confidence = np.mean([p.confidence for p in cluster_preds])
                scd_count = sum(1 for p in cluster_preds if p.class_name == "SCD")
                scd_percentage = scd_count / size
            else:
                avg_confidence = 0.0
                scd_percentage = 0.0

            cluster = Cluster(
                cluster_id=int(label),
                size=size,
                centroid=centroid,
                members=member_indices,
                avg_confidence=avg_confidence,
                scd_percentage=scd_percentage,
            )

            clusters.append(cluster)

        # Sort by SCD percentage (highest priority first)
        clusters.sort(key=lambda c: c.scd_percentage, reverse=True)

        return clusters, labels

    def compute_spatial_statistics(
        self, coordinates: List[Tuple[float, float]], predictions: Optional[List] = None
    ) -> Dict:
        """
        Compute comprehensive spatial statistics.

        Args:
            coordinates: List of (lon, lat) coordinates
            predictions: Optional list of predictions

        Returns:
            Dictionary with spatial statistics
        """
        # Cluster analysis
        clusters, labels = self.cluster_structures(coordinates, predictions)

        # Nearest neighbor distances
        if len(coordinates) > 1:
            dist_matrix = self.compute_distance_matrix(coordinates)
            np.fill_diagonal(dist_matrix, np.inf)
            nearest_neighbor_distances = np.min(dist_matrix, axis=1)
            avg_nn_distance = np.mean(nearest_neighbor_distances)
            median_nn_distance = np.median(nearest_neighbor_distances)
        else:
            avg_nn_distance = 0.0
            median_nn_distance = 0.0
            nearest_neighbor_distances = []

        # Clustering metrics
        n_clusters = len(clusters)
        n_noise = np.sum(labels == -1)
        clustered_percentage = (
            (len(coordinates) - n_noise) / len(coordinates) if len(coordinates) > 0 else 0.0
        )

        stats = {
            "total_structures": len(coordinates),
            "n_clusters": n_clusters,
            "n_isolated": n_noise,
            "clustered_percentage": clustered_percentage,
            "avg_nearest_neighbor_km": avg_nn_distance,
            "median_nearest_neighbor_km": median_nn_distance,
            "clusters": [c.to_dict() for c in clusters],
            "high_priority_clusters": [c.to_dict() for c in clusters if c.is_high_priority()],
        }

        if predictions:
            scd_coords = [
                coord for coord, pred in zip(coordinates, predictions) if pred.class_name == "SCD"
            ]
            stats["scd_count"] = len(scd_coords)
            stats["scd_percentage"] = len(scd_coords) / len(predictions) if predictions else 0.0

        return stats

    def generate_heatmap_grid(
        self,
        coordinates: List[Tuple[float, float]],
        grid_size: int = 50,
        weights: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Generate density heatmap grid.

        Args:
            coordinates: List of (lon, lat) coordinates
            grid_size: Number of grid cells per dimension
            weights: Optional weights for each coordinate (e.g., confidence)

        Returns:
            Tuple of (heatmap array, bounds (minlon, maxlon, minlat, maxlat))
        """
        if len(coordinates) == 0:
            return np.zeros((grid_size, grid_size)), (0, 0, 0, 0)

        coords_array = np.array(coordinates)
        lons = coords_array[:, 0]
        lats = coords_array[:, 1]

        # Define bounds
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()

        # Add padding
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        padding = 0.1

        lon_min -= lon_range * padding
        lon_max += lon_range * padding
        lat_min -= lat_range * padding
        lat_max += lat_range * padding

        # Create heatmap
        heatmap = np.zeros((grid_size, grid_size))

        if weights is None:
            weights = np.ones(len(coordinates))

        for (lon, lat), weight in zip(coordinates, weights):
            # Find grid cell
            x_idx = int((lon - lon_min) / (lon_max - lon_min) * (grid_size - 1))
            y_idx = int((lat - lat_min) / (lat_max - lat_min) * (grid_size - 1))

            # Clamp to grid
            x_idx = max(0, min(grid_size - 1, x_idx))
            y_idx = max(0, min(grid_size - 1, y_idx))

            heatmap[y_idx, x_idx] += weight

        bounds = (lon_min, lon_max, lat_min, lat_max)

        return heatmap, bounds
