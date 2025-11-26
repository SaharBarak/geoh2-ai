"""
Sentinel-2 Data Fetcher

Retrieves multispectral satellite imagery from Sentinel-2.
Supports Sentinel Hub and Google Earth Engine backends.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class Sentinel2Config:
    """Configuration for Sentinel-2 data fetching."""

    provider: str = "sentinel_hub"  # "sentinel_hub" or "earth_engine"
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    resolution: int = 10  # meters
    max_cloud_cover: float = 20  # percent
    bands: List[str] = None
    cache_dir: str = "data/cache/sentinel2"

    def __post_init__(self):
        if self.bands is None:
            self.bands = ["B2", "B3", "B4", "B8", "B11", "B12"]

        # Load from environment if not provided
        if self.client_id is None:
            self.client_id = os.environ.get("SENTINEL_HUB_CLIENT_ID")
        if self.client_secret is None:
            self.client_secret = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")


@dataclass
class Sentinel2Image:
    """Container for Sentinel-2 image data."""

    bands: Dict[str, np.ndarray]
    metadata: Dict
    bbox: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    date: datetime
    cloud_cover: float

    def get_rgb(self) -> np.ndarray:
        """Get RGB image from bands B4, B3, B2."""
        r = self.bands.get("B4", np.zeros((100, 100)))
        g = self.bands.get("B3", np.zeros((100, 100)))
        b = self.bands.get("B2", np.zeros((100, 100)))

        # Stack and normalize
        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.clip(rgb / 3000, 0, 1)  # Typical Sentinel-2 reflectance scale

        return (rgb * 255).astype(np.uint8)

    def get_band(self, band_name: str) -> np.ndarray:
        """Get a specific band."""
        if band_name not in self.bands:
            raise KeyError(
                f"Band {band_name} not available. Available: {list(self.bands.keys())}"
            )
        return self.bands[band_name]


class Sentinel2Fetcher:
    """
    Fetcher for Sentinel-2 satellite imagery.

    Supports multiple backends:
    - Sentinel Hub API
    - Google Earth Engine
    """

    def __init__(self, config: Optional[Sentinel2Config] = None):
        """
        Initialize the fetcher.

        Args:
            config: Configuration for data fetching
        """
        self.config = config or Sentinel2Config()
        self._client = None
        self._cache_dir = Path(self.config.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _init_sentinel_hub(self):
        """Initialize Sentinel Hub client."""
        try:
            from sentinelhub import (
                CRS,
                BBox,
                DataCollection,
                MimeType,
                SentinelHubRequest,
                SHConfig,
            )

            config = SHConfig()
            config.sh_client_id = self.config.client_id
            config.sh_client_secret = self.config.client_secret

            self._sh_config = config
            self._DataCollection = DataCollection
            self._MimeType = MimeType
            self._BBox = BBox
            self._CRS = CRS
            self._SentinelHubRequest = SentinelHubRequest

            return True

        except ImportError:
            print(
                "Sentinel Hub SDK not installed. Install with: pip install sentinelhub"
            )
            return False

    def _init_earth_engine(self):
        """Initialize Earth Engine client."""
        try:
            import ee

            # Try to authenticate
            try:
                ee.Initialize()
            except Exception:
                ee.Authenticate()
                ee.Initialize()

            self._ee = ee
            return True

        except ImportError:
            print(
                "Earth Engine API not installed. Install with: pip install earthengine-api"
            )
            return False

    def fetch(
        self,
        lon: float,
        lat: float,
        width_meters: float = 640,
        height_meters: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[Sentinel2Image]:
        """
        Fetch Sentinel-2 image for a location.

        Args:
            lon: Longitude of center point
            lat: Latitude of center point
            width_meters: Width of area to fetch
            height_meters: Height of area (defaults to width)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Sentinel2Image or None if no data available
        """
        height_meters = height_meters or width_meters

        # Default to last 2 years
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        # Check cache first
        cache_key = f"{lon:.4f}_{lat:.4f}_{width_meters}_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        # Fetch based on provider
        if self.config.provider == "sentinel_hub":
            result = self._fetch_sentinel_hub(
                lon, lat, width_meters, height_meters, start_date, end_date
            )
        else:
            result = self._fetch_earth_engine(
                lon, lat, width_meters, height_meters, start_date, end_date
            )

        # Cache result
        if result is not None:
            self._save_to_cache(cache_key, result)

        return result

    def _fetch_sentinel_hub(
        self,
        lon: float,
        lat: float,
        width_meters: float,
        height_meters: float,
        start_date: str,
        end_date: str,
    ) -> Optional[Sentinel2Image]:
        """Fetch using Sentinel Hub API."""
        if not self._init_sentinel_hub():
            return None

        # Calculate bounding box
        from src.preprocessing.coordinate_handler import CoordinateHandler

        handler = CoordinateHandler()
        bbox = handler.create_bounding_box(lon, lat, width_meters, height_meters)

        # Create evalscript for requested bands
        evalscript = self._create_evalscript(self.config.bands)

        try:
            # Create request
            request = self._SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    self._SentinelHubRequest.input_data(
                        data_collection=self._DataCollection.SENTINEL2_L2A,
                        time_interval=(start_date, end_date),
                        maxcc=self.config.max_cloud_cover / 100,
                    )
                ],
                responses=[
                    self._SentinelHubRequest.output_response(
                        "default", self._MimeType.TIFF
                    )
                ],
                bbox=self._BBox(bbox.to_tuple(), crs=self._CRS.WGS84),
                size=(
                    int(width_meters / self.config.resolution),
                    int(height_meters / self.config.resolution),
                ),
                config=self._sh_config,
            )

            # Execute request
            data = request.get_data()[0]

            # Parse bands
            bands = {}
            for i, band_name in enumerate(self.config.bands):
                if i < data.shape[-1]:
                    bands[band_name] = data[:, :, i].astype(np.float32)

            return Sentinel2Image(
                bands=bands,
                metadata={
                    "provider": "sentinel_hub",
                    "resolution": self.config.resolution,
                },
                bbox=bbox.to_tuple(),
                date=datetime.now(),
                cloud_cover=0,
            )

        except Exception as e:
            print(f"Sentinel Hub fetch error: {e}")
            return None

    def _fetch_earth_engine(
        self,
        lon: float,
        lat: float,
        width_meters: float,
        height_meters: float,
        start_date: str,
        end_date: str,
    ) -> Optional[Sentinel2Image]:
        """Fetch using Google Earth Engine."""
        if not self._init_earth_engine():
            return None

        try:
            ee = self._ee

            # Create point and buffer
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(max(width_meters, height_meters) / 2)

            # Get Sentinel-2 collection
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR")
                .filterBounds(region)
                .filterDate(start_date, end_date)
                .filter(
                    ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.config.max_cloud_cover)
                )
                .sort("CLOUDY_PIXEL_PERCENTAGE")
            )

            # Get best image
            image = collection.first()

            if image is None:
                return None

            # Select bands
            image = image.select(self.config.bands)

            # Get as numpy array
            scale = self.config.resolution
            data = image.sampleRectangle(region, defaultValue=0).getInfo()

            # Parse bands
            bands = {}
            for band_name in self.config.bands:
                if band_name in data["properties"]:
                    bands[band_name] = np.array(
                        data["properties"][band_name], dtype=np.float32
                    )

            # Get metadata
            info = image.getInfo()
            cloud_cover = info.get("properties", {}).get("CLOUDY_PIXEL_PERCENTAGE", 0)
            date_str = info.get("properties", {}).get("system:time_start", 0)
            date = (
                datetime.fromtimestamp(date_str / 1000) if date_str else datetime.now()
            )

            return Sentinel2Image(
                bands=bands,
                metadata={
                    "provider": "earth_engine",
                    "image_id": info.get("id", ""),
                    "resolution": self.config.resolution,
                },
                bbox=(
                    lon - width_meters / 222000,
                    lat - height_meters / 222000,
                    lon + width_meters / 222000,
                    lat + height_meters / 222000,
                ),
                date=date,
                cloud_cover=cloud_cover,
            )

        except Exception as e:
            print(f"Earth Engine fetch error: {e}")
            return None

    def _create_evalscript(self, bands: List[str]) -> str:
        """Create Sentinel Hub evalscript for requested bands."""
        band_inputs = ", ".join([f'"{b}"' for b in bands])
        band_outputs = ", ".join([f"sample.{b}" for b in bands])

        return f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: [{band_inputs}] }}],
    output: {{ bands: {len(bands)}, sampleType: "FLOAT32" }}
  }};
}}

function evaluatePixel(sample) {{
  return [{band_outputs}];
}}
"""

    def _load_from_cache(self, cache_key: str) -> Optional[Sentinel2Image]:
        """Load image from cache."""
        cache_file = self._cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                return Sentinel2Image(
                    bands=dict(data["bands"].item()),
                    metadata=dict(data["metadata"].item()),
                    bbox=tuple(data["bbox"]),
                    date=datetime.fromtimestamp(data["timestamp"]),
                    cloud_cover=float(data["cloud_cover"]),
                )
            except Exception:
                return None

        return None

    def _save_to_cache(self, cache_key: str, image: Sentinel2Image) -> None:
        """Save image to cache."""
        cache_file = self._cache_dir / f"{cache_key}.npz"

        try:
            np.savez(
                cache_file,
                bands=image.bands,
                metadata=image.metadata,
                bbox=np.array(image.bbox),
                timestamp=image.date.timestamp(),
                cloud_cover=image.cloud_cover,
            )
        except Exception as e:
            print(f"Failed to cache image: {e}")

    def fetch_batch(
        self, coordinates: List[Tuple[float, float]], **kwargs
    ) -> List[Optional[Sentinel2Image]]:
        """
        Fetch images for multiple coordinates.

        Args:
            coordinates: List of (lon, lat) tuples
            **kwargs: Additional arguments for fetch()

        Returns:
            List of Sentinel2Image objects (None for failures)
        """
        results = []

        for lon, lat in coordinates:
            try:
                image = self.fetch(lon, lat, **kwargs)
                results.append(image)
            except Exception as e:
                print(f"Failed to fetch ({lon}, {lat}): {e}")
                results.append(None)

        return results
