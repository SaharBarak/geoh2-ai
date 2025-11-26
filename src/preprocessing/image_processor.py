"""
Image Processor - Preprocessing Pipeline

Handles image transformations for model input:
- Normalization
- Resizing
- Format conversion
- Augmentation (training)
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class ImageProcessor:
    """
    Image preprocessing pipeline for H2 seep detection.

    Uses function composition for building preprocessing pipelines.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        normalize: bool = True,
        to_rgb: bool = True,
    ):
        """
        Initialize the image processor.

        Args:
            target_size: Target (height, width) for output
            normalize: Whether to normalize to [0, 1]
            to_rgb: Whether to convert to RGB format
        """
        self.target_size = target_size
        self.normalize = normalize
        self.to_rgb = to_rgb

    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from disk.

        Args:
            path: Path to image file

        Returns:
            Image array (H, W, C) in RGB format
        """
        import cv2

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load image: {path}")

        # BGR to RGB
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def resize(
        self,
        image: np.ndarray,
        size: Optional[Tuple[int, int]] = None,
        interpolation: str = "linear",
    ) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image
            size: Target (height, width), uses default if None
            interpolation: Interpolation method

        Returns:
            Resized image
        """
        import cv2

        size = size or self.target_size

        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }

        interp = interp_map.get(interpolation, cv2.INTER_LINEAR)

        return cv2.resize(image, (size[1], size[0]), interpolation=interp)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.

        Args:
            image: Input image (uint8 or float)

        Returns:
            Normalized float32 image
        """
        img = image.astype(np.float32)

        if img.max() > 1.0:
            img = img / 255.0

        return img

    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image from [0, 1] to [0, 255] uint8.

        Args:
            image: Normalized image

        Returns:
            uint8 image
        """
        img = np.clip(image, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img

    def to_tensor_format(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image from HWC to CHW format.

        Args:
            image: Image in (H, W, C) format

        Returns:
            Image in (C, H, W) format
        """
        if image.ndim == 3:
            return np.transpose(image, (2, 0, 1))
        return image

    def from_tensor_format(self, tensor: np.ndarray) -> np.ndarray:
        """
        Convert tensor from CHW to HWC format.

        Args:
            tensor: Tensor in (C, H, W) format

        Returns:
            Image in (H, W, C) format
        """
        if tensor.ndim == 3:
            return np.transpose(tensor, (1, 2, 0))
        return tensor

    def center_crop(
        self,
        image: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Center crop image to specified size.

        Args:
            image: Input image (H, W, C)
            size: Target (height, width)

        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        th, tw = size

        if h < th or w < tw:
            raise ValueError(f"Image size ({h}, {w}) smaller than crop size ({th}, {tw})")

        y = (h - th) // 2
        x = (w - tw) // 2

        return image[y : y + th, x : x + tw]

    def pad_to_square(
        self,
        image: np.ndarray,
        fill_value: int = 0,
    ) -> np.ndarray:
        """
        Pad image to square with fill value.

        Args:
            image: Input image
            fill_value: Padding fill value

        Returns:
            Square-padded image
        """
        h, w = image.shape[:2]
        target = max(h, w)

        if h == w:
            return image

        # Calculate padding
        pad_h = (target - h) // 2
        pad_w = (target - w) // 2

        if image.ndim == 3:
            result = np.full((target, target, image.shape[2]), fill_value, dtype=image.dtype)
        else:
            result = np.full((target, target), fill_value, dtype=image.dtype)

        result[pad_h : pad_h + h, pad_w : pad_w + w] = image

        return result

    def process(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            image: Image path or array

        Returns:
            Processed image ready for model input
        """
        # Load if path
        if isinstance(image, (str, Path)):
            img = self.load_image(image)
        else:
            img = image.copy()

        # Resize
        img = self.resize(img)

        # Normalize
        if self.normalize:
            img = self.normalize_image(img)

        return img

    def process_for_model(
        self,
        image: Union[str, Path, np.ndarray],
        add_batch_dim: bool = True,
    ) -> np.ndarray:
        """
        Preprocess image for model inference.

        Args:
            image: Image path or array
            add_batch_dim: Whether to add batch dimension

        Returns:
            Model-ready tensor (N, C, H, W) or (C, H, W)
        """
        img = self.process(image)
        img = self.to_tensor_format(img)

        if add_batch_dim:
            img = np.expand_dims(img, axis=0)

        return img

    def process_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
    ) -> np.ndarray:
        """
        Process a batch of images.

        Args:
            images: List of image paths or arrays

        Returns:
            Batch tensor (N, C, H, W)
        """
        processed = []
        for img in images:
            tensor = self.process_for_model(img, add_batch_dim=False)
            processed.append(tensor)

        return np.stack(processed, axis=0)


def compose(*functions: Callable) -> Callable:
    """
    Compose multiple functions into a single function.

    Functions are applied left-to-right (first function receives input).

    Args:
        *functions: Functions to compose

    Returns:
        Composed function
    """

    def composed(x):
        for f in functions:
            x = f(x)
        return x

    return composed


def create_preprocessing_pipeline(
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a preprocessing pipeline function.

    Args:
        target_size: Target image size
        normalize: Whether to normalize

    Returns:
        Preprocessing function
    """
    processor = ImageProcessor(target_size=target_size, normalize=normalize)

    return compose(
        processor.resize,
        processor.normalize_image if normalize else lambda x: x,
        processor.to_tensor_format,
    )
