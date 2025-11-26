"""
Google Maps Scraper

Downloads high-resolution satellite imagery from Google Maps.
Supports API-based and Selenium-based approaches.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class GoogleMapsConfig:
    """Configuration for Google Maps scraping."""

    api_key: Optional[str] = None
    zoom_level: int = 18
    image_size: int = 640
    map_type: str = "satellite"
    use_selenium: bool = False
    headless: bool = True
    requests_per_second: float = 1.0
    max_retries: int = 3
    cache_dir: str = "data/cache/google_maps"

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("GOOGLE_MAPS_API_KEY")


@dataclass
class GoogleMapsImage:
    """Container for Google Maps image."""

    image: np.ndarray
    center: Tuple[float, float]  # (lon, lat)
    zoom: int
    metadata: dict


class GoogleMapsScraper:
    """
    Scraper for Google Maps satellite imagery.

    Provides high-resolution imagery for H2 seep detection
    (achieves 90% accuracy per the paper).
    """

    def __init__(self, config: Optional[GoogleMapsConfig] = None):
        """
        Initialize the scraper.

        Args:
            config: Scraper configuration
        """
        self.config = config or GoogleMapsConfig()
        self._cache_dir = Path(self.config.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0
        self._driver = None

    def _rate_limit(self):
        """Apply rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        min_interval = 1.0 / self.config.requests_per_second

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time = time.time()

    def download(
        self,
        lon: float,
        lat: float,
        zoom: Optional[int] = None,
        size: Optional[int] = None,
    ) -> Optional[GoogleMapsImage]:
        """
        Download satellite image for a location.

        Args:
            lon: Longitude
            lat: Latitude
            zoom: Zoom level (1-21, higher = more detail)
            size: Image size in pixels

        Returns:
            GoogleMapsImage or None if failed
        """
        zoom = zoom or self.config.zoom_level
        size = size or self.config.image_size

        # Check cache
        cache_key = f"{lon:.6f}_{lat:.6f}_{zoom}_{size}"
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        # Apply rate limiting
        self._rate_limit()

        # Try API first, then Selenium
        if self.config.api_key and not self.config.use_selenium:
            image = self._download_api(lon, lat, zoom, size)
        else:
            image = self._download_selenium(lon, lat, zoom, size)

        # Cache result
        if image is not None:
            self._save_to_cache(cache_key, image)

        return image

    def _download_api(
        self,
        lon: float,
        lat: float,
        zoom: int,
        size: int,
    ) -> Optional[GoogleMapsImage]:
        """Download using Google Static Maps API."""
        import requests

        url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": f"{size}x{size}",
            "maptype": self.config.map_type,
            "key": self.config.api_key,
        }

        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                # Parse image
                from PIL import Image
                import io

                img = Image.open(io.BytesIO(response.content))
                img_array = np.array(img)

                return GoogleMapsImage(
                    image=img_array,
                    center=(lon, lat),
                    zoom=zoom,
                    metadata={
                        "source": "google_maps_api",
                        "size": size,
                    },
                )

            except Exception as e:
                print(f"API download attempt {attempt + 1} failed: {e}")
                time.sleep(1)

        return None

    def _download_selenium(
        self,
        lon: float,
        lat: float,
        zoom: int,
        size: int,
    ) -> Optional[GoogleMapsImage]:
        """Download using Selenium browser automation."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            from PIL import Image
            import io

            # Initialize driver if needed
            if self._driver is None:
                options = Options()
                if self.config.headless:
                    options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument(f"--window-size={size},{size}")

                service = Service(ChromeDriverManager().install())
                self._driver = webdriver.Chrome(service=service, options=options)

            # Navigate to Google Maps
            url = f"https://www.google.com/maps/@{lat},{lon},{zoom}z/data=!3m1!1e3"
            self._driver.get(url)

            # Wait for tiles to load
            time.sleep(3)

            # Take screenshot
            screenshot = self._driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(screenshot))
            img_array = np.array(img)

            # Crop to square if needed
            h, w = img_array.shape[:2]
            min_dim = min(h, w)
            start_h = (h - min_dim) // 2
            start_w = (w - min_dim) // 2
            img_array = img_array[
                start_h : start_h + min_dim, start_w : start_w + min_dim
            ]

            # Resize if needed
            if img_array.shape[0] != size:
                import cv2

                img_array = cv2.resize(img_array, (size, size))

            return GoogleMapsImage(
                image=img_array,
                center=(lon, lat),
                zoom=zoom,
                metadata={
                    "source": "selenium",
                    "size": size,
                },
            )

        except Exception as e:
            print(f"Selenium download failed: {e}")
            return None

    def download_batch(
        self, coordinates: List[Tuple[float, float]], **kwargs
    ) -> List[Optional[GoogleMapsImage]]:
        """
        Download images for multiple coordinates.

        Args:
            coordinates: List of (lon, lat) tuples
            **kwargs: Additional arguments for download()

        Returns:
            List of GoogleMapsImage objects (None for failures)
        """
        results = []

        for i, (lon, lat) in enumerate(coordinates):
            print(f"Downloading {i+1}/{len(coordinates)}: ({lon:.4f}, {lat:.4f})")

            try:
                image = self.download(lon, lat, **kwargs)
                results.append(image)
            except Exception as e:
                print(f"Failed: {e}")
                results.append(None)

        return results

    def _load_from_cache(self, cache_key: str) -> Optional[GoogleMapsImage]:
        """Load image from cache."""
        cache_file = self._cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                return GoogleMapsImage(
                    image=data["image"],
                    center=tuple(data["center"]),
                    zoom=int(data["zoom"]),
                    metadata=dict(data["metadata"].item()),
                )
            except Exception:
                return None

        return None

    def _save_to_cache(self, cache_key: str, image: GoogleMapsImage) -> None:
        """Save image to cache."""
        cache_file = self._cache_dir / f"{cache_key}.npz"

        try:
            np.savez(
                cache_file,
                image=image.image,
                center=np.array(image.center),
                zoom=image.zoom,
                metadata=image.metadata,
            )
        except Exception as e:
            print(f"Failed to cache image: {e}")

    def close(self):
        """Close browser driver if open."""
        if self._driver is not None:
            self._driver.quit()
            self._driver = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
