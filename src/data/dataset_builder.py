"""
Dataset Builder for H2 Seep Detection
Constructs training/validation datasets from imagery
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image


@dataclass
class DatasetConfig:
    """Configuration for dataset building"""
    data_dir: Path
    train_split: float = 0.8
    val_split: float = 0.2
    image_size: int = 640
    seed: int = 42


class DatasetBuilder:
    """
    Builds YOLO-compatible classification datasets.
    Organizes images into class-based directory structure.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        np.random.seed(config.seed)

    def create_directory_structure(
        self,
        class_names: List[str]
    ) -> Dict[str, Path]:
        """
        Create dataset directory structure.

        Args:
            class_names: List of class names

        Returns:
            Dictionary of created directories
        """
        dirs = {}

        for split in ["train", "val"]:
            split_dir = self.data_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for class_name in class_names:
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)
                dirs[f"{split}/{class_name}"] = class_dir

        return dirs

    def split_dataset(
        self,
        image_paths: List[Path],
        labels: List[int],
        stratify: bool = True
    ) -> Tuple[List[Path], List[int], List[Path], List[int]]:
        """
        Split dataset into train/val sets.

        Args:
            image_paths: List of image paths
            labels: List of class labels
            stratify: Whether to stratify by class

        Returns:
            Tuple of (train_paths, train_labels, val_paths, val_labels)
        """
        if stratify:
            # Stratified split per class
            train_paths, train_labels = [], []
            val_paths, val_labels = [], []

            unique_labels = np.unique(labels)
            for label in unique_labels:
                # Get indices for this class
                indices = [i for i, l in enumerate(labels) if l == label]
                n_train = int(len(indices) * self.config.train_split)

                # Shuffle and split
                np.random.shuffle(indices)
                train_idx = indices[:n_train]
                val_idx = indices[n_train:]

                train_paths.extend([image_paths[i] for i in train_idx])
                train_labels.extend([labels[i] for i in train_idx])
                val_paths.extend([image_paths[i] for i in val_idx])
                val_labels.extend([labels[i] for i in val_idx])
        else:
            # Random split
            indices = np.arange(len(image_paths))
            np.random.shuffle(indices)
            n_train = int(len(indices) * self.config.train_split)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

            train_paths = [image_paths[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_paths = [image_paths[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

        return train_paths, train_labels, val_paths, val_labels

    def create_dataset_yaml(
        self,
        class_names: List[str],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create YAML configuration file for YOLO training.

        Args:
            class_names: List of class names
            output_path: Where to save YAML file

        Returns:
            Path to created YAML file
        """
        if output_path is None:
            output_path = self.data_dir / "dataset.yaml"

        dataset_config = {
            "path": str(self.data_dir.absolute()),
            "train": "train",
            "val": "val",
            "names": {i: name for i, name in enumerate(class_names)},
            "nc": len(class_names)
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        return output_path

    def get_dataset_statistics(self) -> Dict:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "train": {},
            "val": {},
            "total": {}
        }

        for split in ["train", "val"]:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue

            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    n_images = len(list(class_dir.glob("*.jpg"))) + \
                               len(list(class_dir.glob("*.png")))
                    stats[split][class_name] = n_images

                    if class_name not in stats["total"]:
                        stats["total"][class_name] = 0
                    stats["total"][class_name] += n_images

        return stats

    def validate_dataset(self) -> bool:
        """
        Validate dataset structure and completeness.

        Returns:
            True if dataset is valid
        """
        # Check train and val directories exist
        if not (self.data_dir / "train").exists():
            print("Error: train directory not found")
            return False

        if not (self.data_dir / "val").exists():
            print("Error: val directory not found")
            return False

        # Check for images
        stats = self.get_dataset_statistics()
        if not stats["train"] or not stats["val"]:
            print("Error: No images found in dataset")
            return False

        print("Dataset validation successful!")
        print(f"Train classes: {len(stats['train'])}")
        print(f"Val classes: {len(stats['val'])}")
        print(f"Total images: {sum(stats['total'].values())}")

        return True
