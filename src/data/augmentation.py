"""
Data Augmentation Pipeline

Provides augmentation transforms for training data.
Based on Albumentations library for efficient transformations.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    rotation_limit: int = 45
    rotation_prob: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    brightness_contrast_prob: float = 0.3
    hue_shift: int = 10
    saturation_shift: int = 20
    color_jitter_prob: float = 0.2
    blur_limit: int = 3
    blur_prob: float = 0.1
    noise_var: float = 0.01
    noise_prob: float = 0.1


class Augmentor:
    """
    Data augmentation pipeline for H2 seep detection.

    Provides both Albumentations-based and NumPy-based
    augmentation transforms.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmentor.

        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        self._albumentations_available = self._check_albumentations()
        self._transform = None

    def _check_albumentations(self) -> bool:
        """Check if Albumentations is available."""
        try:
            import albumentations
            return True
        except ImportError:
            return False

    def get_transform(self) -> Callable:
        """
        Get the augmentation transform.

        Returns Albumentations transform if available,
        otherwise returns NumPy-based transform.
        """
        if self._transform is not None:
            return self._transform

        if self._albumentations_available:
            self._transform = self._create_albumentations_transform()
        else:
            self._transform = self._create_numpy_transform()

        return self._transform

    def _create_albumentations_transform(self) -> Callable:
        """Create Albumentations transform pipeline."""
        import albumentations as A

        transforms = []

        # Flips
        if self.config.horizontal_flip > 0:
            transforms.append(A.HorizontalFlip(p=self.config.horizontal_flip))

        if self.config.vertical_flip > 0:
            transforms.append(A.VerticalFlip(p=self.config.vertical_flip))

        # Rotation
        if self.config.rotation_prob > 0:
            transforms.append(A.Rotate(
                limit=self.config.rotation_limit,
                p=self.config.rotation_prob,
                border_mode=0,  # cv2.BORDER_CONSTANT
            ))

        # Brightness/Contrast
        if self.config.brightness_contrast_prob > 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=self.config.brightness_limit,
                contrast_limit=self.config.contrast_limit,
                p=self.config.brightness_contrast_prob,
            ))

        # Color jitter
        if self.config.color_jitter_prob > 0:
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=self.config.hue_shift,
                sat_shift_limit=self.config.saturation_shift,
                val_shift_limit=20,
                p=self.config.color_jitter_prob,
            ))

        # Blur
        if self.config.blur_prob > 0:
            transforms.append(A.GaussianBlur(
                blur_limit=self.config.blur_limit,
                p=self.config.blur_prob,
            ))

        # Noise
        if self.config.noise_prob > 0:
            transforms.append(A.GaussNoise(
                var_limit=(0, self.config.noise_var * 255 * 255),
                p=self.config.noise_prob,
            ))

        transform = A.Compose(transforms)

        def apply_transform(image: np.ndarray) -> np.ndarray:
            return transform(image=image)["image"]

        return apply_transform

    def _create_numpy_transform(self) -> Callable:
        """Create NumPy-based transform pipeline (fallback)."""
        def apply_transform(image: np.ndarray) -> np.ndarray:
            img = image.copy()

            # Horizontal flip
            if np.random.random() < self.config.horizontal_flip:
                img = np.fliplr(img)

            # Vertical flip
            if np.random.random() < self.config.vertical_flip:
                img = np.flipud(img)

            # Rotation (simple 90 degree rotations)
            if np.random.random() < self.config.rotation_prob:
                k = np.random.randint(0, 4)
                img = np.rot90(img, k)

            # Brightness
            if np.random.random() < self.config.brightness_contrast_prob:
                factor = 1.0 + np.random.uniform(
                    -self.config.brightness_limit,
                    self.config.brightness_limit
                )
                img = np.clip(img * factor, 0, 255).astype(np.uint8)

            # Noise
            if np.random.random() < self.config.noise_prob:
                noise = np.random.normal(0, self.config.noise_var * 255, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)

            return img

        return apply_transform

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to an image.

        Args:
            image: Input image (H, W, C) or (H, W)

        Returns:
            Augmented image
        """
        transform = self.get_transform()
        return transform(image)

    def augment_batch(
        self,
        images: List[np.ndarray],
        n_augmentations: int = 1,
    ) -> List[np.ndarray]:
        """
        Augment a batch of images.

        Args:
            images: List of input images
            n_augmentations: Number of augmented versions per image

        Returns:
            List of augmented images
        """
        results = []
        transform = self.get_transform()

        for image in images:
            for _ in range(n_augmentations):
                augmented = transform(image)
                results.append(augmented)

        return results


def create_training_augmentor() -> Augmentor:
    """Create augmentor with training-time settings."""
    config = AugmentationConfig(
        horizontal_flip=0.5,
        vertical_flip=0.5,
        rotation_limit=45,
        rotation_prob=0.5,
        brightness_contrast_prob=0.3,
        color_jitter_prob=0.2,
    )
    return Augmentor(config)


def create_validation_augmentor() -> Augmentor:
    """Create augmentor with minimal transforms (for TTA)."""
    config = AugmentationConfig(
        horizontal_flip=0.5,
        vertical_flip=0.0,
        rotation_prob=0.0,
        brightness_contrast_prob=0.0,
        color_jitter_prob=0.0,
    )
    return Augmentor(config)


class TestTimeAugmentation:
    """
    Test-time augmentation for improved predictions.

    Applies multiple augmentations and aggregates predictions.
    """

    def __init__(self, n_augmentations: int = 4):
        """
        Initialize TTA.

        Args:
            n_augmentations: Number of augmented versions
        """
        self.n_augmentations = n_augmentations
        self.augmentor = create_validation_augmentor()

    def __call__(
        self,
        model,
        image: np.ndarray,
    ) -> Dict[str, float]:
        """
        Apply TTA and return aggregated predictions.

        Args:
            model: Model with predict method
            image: Input image

        Returns:
            Aggregated probability dictionary
        """
        all_probs = []

        # Original image
        result = model.predict(image)
        all_probs.append(list(result.probabilities.values()))

        # Augmented versions
        for _ in range(self.n_augmentations - 1):
            aug_image = self.augmentor(image)
            result = model.predict(aug_image)
            all_probs.append(list(result.probabilities.values()))

        # Aggregate
        avg_probs = np.mean(all_probs, axis=0)

        # Build result dictionary
        class_names = list(result.probabilities.keys())
        return {name: float(prob) for name, prob in zip(class_names, avg_probs)}
