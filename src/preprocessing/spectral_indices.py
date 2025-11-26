"""
Spectral Index Calculator - Key Scientific Contribution

Implements spectral indices from Ginzburg et al. (2025) for
detecting H2-related sub-circular depressions from Sentinel-2 data.

Primary indices:
- NDVI: Vegetation anomalies (main indicator)
- BI: Brightness Index (secondary indicator)

The paper achieved 70% accuracy using NDVI + BI from Sentinel-2.
"""

from typing import Callable, Dict, List, Optional, Union

import numpy as np

from . import IndexResult, INDEX_DEFINITIONS


class SpectralIndexCalculator:
    """
    Calculator for spectral indices from multispectral satellite data.

    Implements Strategy pattern with pluggable index calculations.
    Designed for Sentinel-2 data but adaptable to other sensors.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize the calculator.

        Args:
            epsilon: Small value to prevent division by zero
        """
        self.epsilon = epsilon

        # Register index calculation functions (Strategy pattern)
        self._indices: Dict[str, Callable] = {
            "ndvi": self.ndvi,
            "bi": self.brightness_index,
            "ndwi": self.ndwi,
            "savi": self.savi,
            "evi": self.evi,
            "ndmi": self.ndmi,
            "si": self.salinity_index,
            "ndsi": self.ndsi,
            "nbr": self.nbr,
        }

    @property
    def available_indices(self) -> List[str]:
        """List of available spectral indices."""
        return list(self._indices.keys())

    def _safe_divide(
        self, numerator: np.ndarray, denominator: np.ndarray
    ) -> np.ndarray:
        """
        Safe division handling zero denominators.

        Args:
            numerator: Numerator array
            denominator: Denominator array

        Returns:
            Result with NaN where denominator is zero
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(
                np.abs(denominator) > self.epsilon, numerator / denominator, np.nan
            )
        return result

    def _validate_band(self, band: np.ndarray, name: str) -> np.ndarray:
        """
        Validate and normalize a spectral band.

        Args:
            band: Input band array
            name: Band name for error messages

        Returns:
            Validated band array
        """
        band = np.asarray(band, dtype=np.float32)

        if band.size == 0:
            raise ValueError(f"Band {name} is empty")

        # Normalize to 0-1 if in DN range
        if band.max() > 1.0:
            band = band / 10000.0  # Sentinel-2 scaling factor

        return band

    def compute(
        self, index_name: str, bands: Dict[str, np.ndarray], **kwargs
    ) -> IndexResult:
        """
        Compute a spectral index.

        Args:
            index_name: Name of the index (e.g., "ndvi", "bi")
            bands: Dictionary of band arrays {band_name: array}
            **kwargs: Additional arguments for specific indices

        Returns:
            IndexResult with computed values
        """
        index_name = index_name.lower()

        if index_name not in self._indices:
            raise ValueError(
                f"Unknown index: {index_name}. " f"Available: {self.available_indices}"
            )

        # Get the calculation function
        calc_func = self._indices[index_name]

        # Call with bands
        return calc_func(bands=bands, **kwargs)

    def compute_multiple(
        self, index_names: List[str], bands: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, IndexResult]:
        """
        Compute multiple spectral indices.

        Args:
            index_names: List of index names
            bands: Dictionary of band arrays
            **kwargs: Additional arguments

        Returns:
            Dictionary of index results
        """
        results = {}
        for name in index_names:
            try:
                results[name] = self.compute(name, bands, **kwargs)
            except KeyError as e:
                # Missing band, skip this index
                continue
        return results

    def compute_all(
        self, bands: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, IndexResult]:
        """
        Compute all available indices.

        Args:
            bands: Dictionary of band arrays
            **kwargs: Additional arguments

        Returns:
            Dictionary of all successfully computed indices
        """
        return self.compute_multiple(self.available_indices, bands, **kwargs)

    # ================================================================
    # Primary Indices (from paper)
    # ================================================================

    def ndvi(
        self,
        nir: Optional[np.ndarray] = None,
        red: Optional[np.ndarray] = None,
        bands: Optional[Dict[str, np.ndarray]] = None,
    ) -> IndexResult:
        """
        Normalized Difference Vegetation Index.

        NDVI = (NIR - RED) / (NIR + RED)

        Primary index for H2 seep detection. Vegetation anomalies
        around seeps show distinctive NDVI patterns.

        Args:
            nir: NIR band array (B8)
            red: RED band array (B4)
            bands: Alternative dict with "B8" and "B4" keys

        Returns:
            IndexResult with NDVI values in [-1, 1]
        """
        if bands is not None:
            nir = bands.get("B8", bands.get("nir", bands.get("NIR")))
            red = bands.get("B4", bands.get("red", bands.get("RED")))

        if nir is None or red is None:
            raise ValueError("NDVI requires NIR (B8) and RED (B4) bands")

        nir = self._validate_band(nir, "NIR")
        red = self._validate_band(red, "RED")

        numerator = nir - red
        denominator = nir + red + self.epsilon

        ndvi_values = self._safe_divide(numerator, denominator)

        return IndexResult(
            name="NDVI",
            value=ndvi_values,
            valid_range=(-1, 1),
            metadata={"formula": "(NIR - RED) / (NIR + RED)"},
        )

    def brightness_index(
        self,
        red: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
        bands: Optional[Dict[str, np.ndarray]] = None,
    ) -> IndexResult:
        """
        Brightness Index.

        BI = sqrt((RED^2 + NIR^2) / 2)

        Secondary index for H2 seep detection. Helps distinguish
        SCDs from salt lakes and other bright features.

        Args:
            red: RED band array (B4)
            nir: NIR band array (B8)
            bands: Alternative dict with band arrays

        Returns:
            IndexResult with BI values in [0, 1]
        """
        if bands is not None:
            red = bands.get("B4", bands.get("red", bands.get("RED")))
            nir = bands.get("B8", bands.get("nir", bands.get("NIR")))

        if red is None or nir is None:
            raise ValueError("BI requires RED (B4) and NIR (B8) bands")

        red = self._validate_band(red, "RED")
        nir = self._validate_band(nir, "NIR")

        bi_values = np.sqrt((red**2 + nir**2) / 2)

        return IndexResult(
            name="BI",
            value=bi_values,
            valid_range=(0, 1),
            metadata={"formula": "sqrt((RED^2 + NIR^2) / 2)"},
        )

    # ================================================================
    # Secondary Indices (supplementary)
    # ================================================================

    def ndwi(
        self,
        green: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
        bands: Optional[Dict[str, np.ndarray]] = None,
    ) -> IndexResult:
        """
        Normalized Difference Water Index.

        NDWI = (GREEN - NIR) / (GREEN + NIR)

        Useful for detecting water bodies and flooding.

        Args:
            green: GREEN band array (B3)
            nir: NIR band array (B8)
            bands: Alternative dict with band arrays

        Returns:
            IndexResult with NDWI values in [-1, 1]
        """
        if bands is not None:
            green = bands.get("B3", bands.get("green", bands.get("GREEN")))
            nir = bands.get("B8", bands.get("nir", bands.get("NIR")))

        if green is None or nir is None:
            raise ValueError("NDWI requires GREEN (B3) and NIR (B8) bands")

        green = self._validate_band(green, "GREEN")
        nir = self._validate_band(nir, "NIR")

        numerator = green - nir
        denominator = green + nir + self.epsilon

        ndwi_values = self._safe_divide(numerator, denominator)

        return IndexResult(
            name="NDWI",
            value=ndwi_values,
            valid_range=(-1, 1),
            metadata={"formula": "(GREEN - NIR) / (GREEN + NIR)"},
        )

    def savi(
        self,
        nir: Optional[np.ndarray] = None,
        red: Optional[np.ndarray] = None,
        bands: Optional[Dict[str, np.ndarray]] = None,
        L: float = 0.5,
    ) -> IndexResult:
        """
        Soil Adjusted Vegetation Index.

        SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)

        Reduces soil background influence on vegetation signal.

        Args:
            nir: NIR band array (B8)
            red: RED band array (B4)
            bands: Alternative dict with band arrays
            L: Soil brightness correction factor (default 0.5)

        Returns:
            IndexResult with SAVI values
        """
        if bands is not None:
            nir = bands.get("B8", bands.get("nir", bands.get("NIR")))
            red = bands.get("B4", bands.get("red", bands.get("RED")))

        if nir is None or red is None:
            raise ValueError("SAVI requires NIR (B8) and RED (B4) bands")

        nir = self._validate_band(nir, "NIR")
        red = self._validate_band(red, "RED")

        numerator = nir - red
        denominator = nir + red + L + self.epsilon

        savi_values = self._safe_divide(numerator, denominator) * (1 + L)

        return IndexResult(
            name="SAVI",
            value=savi_values,
            valid_range=(-1, 1),
            metadata={"formula": f"((NIR - RED) / (NIR + RED + {L})) * (1 + {L})"},
        )

    def evi(
        self,
        nir: Optional[np.ndarray] = None,
        red: Optional[np.ndarray] = None,
        blue: Optional[np.ndarray] = None,
        bands: Optional[Dict[str, np.ndarray]] = None,
        G: float = 2.5,
        C1: float = 6.0,
        C2: float = 7.5,
        L: float = 1.0,
    ) -> IndexResult:
        """
        Enhanced Vegetation Index.

        EVI = G * (NIR - RED) / (NIR + C1*RED - C2*BLUE + L)

        Enhanced vegetation detection with atmospheric correction.

        Args:
            nir: NIR band array (B8)
            red: RED band array (B4)
            blue: BLUE band array (B2)
            bands: Alternative dict with band arrays
            G, C1, C2, L: EVI coefficients

        Returns:
            IndexResult with EVI values
        """
        if bands is not None:
            nir = bands.get("B8", bands.get("nir", bands.get("NIR")))
            red = bands.get("B4", bands.get("red", bands.get("RED")))
            blue = bands.get("B2", bands.get("blue", bands.get("BLUE")))

        if nir is None or red is None or blue is None:
            raise ValueError("EVI requires NIR (B8), RED (B4), and BLUE (B2) bands")

        nir = self._validate_band(nir, "NIR")
        red = self._validate_band(red, "RED")
        blue = self._validate_band(blue, "BLUE")

        numerator = G * (nir - red)
        denominator = nir + C1 * red - C2 * blue + L + self.epsilon

        evi_values = self._safe_divide(numerator, denominator)

        return IndexResult(
            name="EVI",
            value=evi_values,
            valid_range=(-1, 1),
            metadata={"formula": "2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)"},
        )

    def ndmi(
        self,
        nir: Optional[np.ndarray] = None,
        swir: Optional[np.ndarray] = None,
        bands: Optional[Dict[str, np.ndarray]] = None,
    ) -> IndexResult:
        """
        Normalized Difference Moisture Index.

        NDMI = (NIR - SWIR) / (NIR + SWIR)

        Measures vegetation water content.

        Args:
            nir: NIR band array (B8)
            swir: SWIR band array (B11)
            bands: Alternative dict with band arrays

        Returns:
            IndexResult with NDMI values in [-1, 1]
        """
        if bands is not None:
            nir = bands.get("B8", bands.get("nir", bands.get("NIR")))
            swir = bands.get("B11", bands.get("swir", bands.get("SWIR")))

        if nir is None or swir is None:
            raise ValueError("NDMI requires NIR (B8) and SWIR (B11) bands")

        nir = self._validate_band(nir, "NIR")
        swir = self._validate_band(swir, "SWIR")

        numerator = nir - swir
        denominator = nir + swir + self.epsilon

        ndmi_values = self._safe_divide(numerator, denominator)

        return IndexResult(
            name="NDMI",
            value=ndmi_values,
            valid_range=(-1, 1),
            metadata={"formula": "(NIR - SWIR) / (NIR + SWIR)"},
        )

    def salinity_index(
        self,
        green: Optional[np.ndarray] = None,
        red: Optional[np.ndarray] = None,
        bands: Optional[Dict[str, np.ndarray]] = None,
    ) -> IndexResult:
        """
        Salinity Index.

        SI = sqrt(GREEN * RED)

        Detects salt-affected areas.

        Args:
            green: GREEN band array (B3)
            red: RED band array (B4)
            bands: Alternative dict with band arrays

        Returns:
            IndexResult with SI values in [0, 1]
        """
        if bands is not None:
            green = bands.get("B3", bands.get("green", bands.get("GREEN")))
            red = bands.get("B4", bands.get("red", bands.get("RED")))

        if green is None or red is None:
            raise ValueError("SI requires GREEN (B3) and RED (B4) bands")

        green = self._validate_band(green, "GREEN")
        red = self._validate_band(red, "RED")

        si_values = np.sqrt(green * red)

        return IndexResult(
            name="SI",
            value=si_values,
            valid_range=(0, 1),
            metadata={"formula": "sqrt(GREEN * RED)"},
        )

    def ndsi(
        self,
        red: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
        bands: Optional[Dict[str, np.ndarray]] = None,
    ) -> IndexResult:
        """
        Normalized Difference Salinity Index.

        NDSI = (RED - NIR) / (RED + NIR)

        Normalized measure of salinity (opposite of NDVI).

        Args:
            red: RED band array (B4)
            nir: NIR band array (B8)
            bands: Alternative dict with band arrays

        Returns:
            IndexResult with NDSI values in [-1, 1]
        """
        if bands is not None:
            red = bands.get("B4", bands.get("red", bands.get("RED")))
            nir = bands.get("B8", bands.get("nir", bands.get("NIR")))

        if red is None or nir is None:
            raise ValueError("NDSI requires RED (B4) and NIR (B8) bands")

        red = self._validate_band(red, "RED")
        nir = self._validate_band(nir, "NIR")

        numerator = red - nir
        denominator = red + nir + self.epsilon

        ndsi_values = self._safe_divide(numerator, denominator)

        return IndexResult(
            name="NDSI",
            value=ndsi_values,
            valid_range=(-1, 1),
            metadata={"formula": "(RED - NIR) / (RED + NIR)"},
        )

    def nbr(
        self,
        nir: Optional[np.ndarray] = None,
        swir2: Optional[np.ndarray] = None,
        bands: Optional[Dict[str, np.ndarray]] = None,
    ) -> IndexResult:
        """
        Normalized Burn Ratio.

        NBR = (NIR - SWIR2) / (NIR + SWIR2)

        Burn severity and vegetation dryness.

        Args:
            nir: NIR band array (B8)
            swir2: SWIR2 band array (B12)
            bands: Alternative dict with band arrays

        Returns:
            IndexResult with NBR values in [-1, 1]
        """
        if bands is not None:
            nir = bands.get("B8", bands.get("nir", bands.get("NIR")))
            swir2 = bands.get("B12", bands.get("swir2", bands.get("SWIR2")))

        if nir is None or swir2 is None:
            raise ValueError("NBR requires NIR (B8) and SWIR2 (B12) bands")

        nir = self._validate_band(nir, "NIR")
        swir2 = self._validate_band(swir2, "SWIR2")

        numerator = nir - swir2
        denominator = nir + swir2 + self.epsilon

        nbr_values = self._safe_divide(numerator, denominator)

        return IndexResult(
            name="NBR",
            value=nbr_values,
            valid_range=(-1, 1),
            metadata={"formula": "(NIR - SWIR2) / (NIR + SWIR2)"},
        )

    # ================================================================
    # Custom Index Registration
    # ================================================================

    def register_index(
        self,
        name: str,
        func: Callable[..., IndexResult],
    ) -> None:
        """
        Register a custom spectral index.

        Args:
            name: Index name (lowercase)
            func: Calculation function returning IndexResult
        """
        self._indices[name.lower()] = func
