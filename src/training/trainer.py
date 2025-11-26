"""
Trainer - Training Orchestrator

Orchestrates model training with callbacks and validation.
Uses Observer pattern for training event notifications.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class TrainingConfig:
    """Configuration for training."""

    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    patience: int = 10
    optimizer: str = "adamw"
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    amp: bool = True
    save_dir: str = "runs/train"
    device: str = "auto"
    workers: int = 8
    seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        training = config.get("training", {})
        return cls(
            epochs=training.get("epochs", 50),
            batch_size=training.get("batch_size", 16),
            learning_rate=training.get("learning_rate", 0.001),
            patience=training.get("patience", 10),
            optimizer=training.get("optimizer", "adamw"),
            weight_decay=training.get("weight_decay", 0.0005),
            warmup_epochs=training.get("warmup_epochs", 3),
            amp=training.get("amp", True),
            save_dir=config.get("checkpointing", {}).get("save_dir", "runs/train"),
            device=config.get("hardware", {}).get("device", "auto"),
            workers=config.get("dataset", {}).get("workers", 8),
            seed=config.get("reproducibility", {}).get("seed", 42),
        )


@dataclass
class TrainingResult:
    """Result from training run."""

    best_accuracy: float
    final_accuracy: float
    epochs_trained: int
    best_epoch: int
    best_weights_path: str
    last_weights_path: str
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    save_dir: str = ""


class Trainer:
    """
    Training orchestrator for H2 seep detection models.

    Uses Observer pattern to notify callbacks of training events.
    """

    def __init__(
        self,
        model,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train (YOLOv8Classifier or similar)
            config: Training configuration
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.callbacks: List = []
        self._is_training = False
        self._current_epoch = 0
        self._best_metric = 0.0
        self._best_epoch = 0

    def add_callback(self, callback) -> None:
        """Add a training callback."""
        callback.set_trainer(self)
        self.callbacks.append(callback)

    def remove_callback(self, callback) -> None:
        """Remove a training callback."""
        self.callbacks.remove(callback)

    def _notify(self, event: str, data: Optional[Dict] = None) -> None:
        """Notify all callbacks of an event."""
        data = data or {}
        for callback in self.callbacks:
            method = getattr(callback, f"on_{event}", None)
            if method:
                method(data)

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)

        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def train(
        self,
        data_yaml: str,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> TrainingResult:
        """
        Train the model.

        Args:
            data_yaml: Path to dataset YAML file
            epochs: Override epochs from config
            batch_size: Override batch size from config
            **kwargs: Additional training arguments

        Returns:
            TrainingResult with training metrics and paths
        """
        # Set seed
        self._set_seed(self.config.seed)

        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size

        # Notify training start
        self._is_training = True
        self._notify(
            "train_begin",
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "config": self.config,
            },
        )

        try:
            # Use model's built-in training (YOLOv8)
            results = self.model.train_model(
                data_yaml=data_yaml,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=self.config.learning_rate,
                patience=self.config.patience,
                save_dir=self.config.save_dir,
                **kwargs,
            )

            # Extract metrics
            best_accuracy = results.get("metrics", {}).get("accuracy", 0)
            best_weights = results.get("best_weights", "")
            last_weights = results.get("last_weights", "")
            save_dir = results.get("save_dir", self.config.save_dir)

            training_result = TrainingResult(
                best_accuracy=best_accuracy,
                final_accuracy=best_accuracy,
                epochs_trained=epochs,
                best_epoch=epochs,  # YOLO doesn't expose this directly
                best_weights_path=best_weights,
                last_weights_path=last_weights,
                metrics_history={},
                save_dir=save_dir,
            )

            # Notify training end
            self._notify(
                "train_end",
                {
                    "result": training_result,
                },
            )

            return training_result

        except Exception as e:
            self._notify("train_error", {"error": str(e)})
            raise

        finally:
            self._is_training = False

    def train_custom(
        self,
        train_loader,
        val_loader,
        epochs: Optional[int] = None,
    ) -> TrainingResult:
        """
        Custom training loop (for non-YOLO models).

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs

        Returns:
            TrainingResult
        """
        import torch

        epochs = epochs or self.config.epochs

        self._is_training = True
        self._current_epoch = 0
        self._best_metric = 0.0
        self._best_epoch = 0

        metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        # Setup
        device = self._resolve_device()
        model = self.model._model.to(device)
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)
        criterion = torch.nn.CrossEntropyLoss()

        # Notify start
        self._notify("train_begin", {"epochs": epochs})

        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            for epoch in range(epochs):
                self._current_epoch = epoch

                # Notify epoch start
                self._notify("epoch_begin", {"epoch": epoch})

                # Training phase
                model.train()
                train_loss = 0.0
                num_batches = 0

                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    num_batches += 1

                    # Notify batch end
                    self._notify(
                        "batch_end",
                        {
                            "batch": batch_idx,
                            "loss": loss.item(),
                        },
                    )

                avg_train_loss = train_loss / max(num_batches, 1)
                metrics_history["train_loss"].append(avg_train_loss)

                # Validation phase
                val_loss, val_accuracy = self._validate_epoch(
                    model, val_loader, criterion, device
                )
                metrics_history["val_loss"].append(val_loss)
                metrics_history["val_accuracy"].append(val_accuracy)

                # Update best
                if val_accuracy > self._best_metric:
                    self._best_metric = val_accuracy
                    self._best_epoch = epoch
                    torch.save(model.state_dict(), save_dir / "best.pt")

                # Scheduler step
                if scheduler:
                    scheduler.step()

                # Notify epoch end
                self._notify(
                    "epoch_end",
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                    },
                )

                # Check early stopping
                if self._should_stop():
                    break

            # Save last model
            torch.save(model.state_dict(), save_dir / "last.pt")

            result = TrainingResult(
                best_accuracy=self._best_metric,
                final_accuracy=metrics_history["val_accuracy"][-1],
                epochs_trained=self._current_epoch + 1,
                best_epoch=self._best_epoch,
                best_weights_path=str(save_dir / "best.pt"),
                last_weights_path=str(save_dir / "last.pt"),
                metrics_history=metrics_history,
                save_dir=str(save_dir),
            )

            self._notify("train_end", {"result": result})
            return result

        finally:
            self._is_training = False

    def _validate_epoch(
        self,
        model,
        val_loader,
        criterion,
        device,
    ) -> tuple:
        """Run validation for one epoch."""
        import torch

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = val_loss / max(len(val_loader), 1)
        accuracy = correct / max(total, 1)

        return avg_loss, accuracy

    def _resolve_device(self) -> str:
        """Resolve device to use."""
        import torch

        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.config.device

    def _create_optimizer(self, model):
        """Create optimizer based on config."""
        import torch.optim as optim

        params = model.parameters()

        if self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.937,
                weight_decay=self.config.weight_decay,
            )
        else:
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
            )

    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        import torch.optim.lr_scheduler as lr_scheduler

        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
        )

    def _should_stop(self) -> bool:
        """Check if training should stop (early stopping)."""
        for callback in self.callbacks:
            if hasattr(callback, "should_stop") and callback.should_stop:
                return True
        return False

    @property
    def is_training(self) -> bool:
        """Whether training is in progress."""
        return self._is_training

    @property
    def current_epoch(self) -> int:
        """Current epoch number."""
        return self._current_epoch
