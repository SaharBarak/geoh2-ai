"""
Training Callbacks - Observer Pattern Implementation

Provides hooks into the training loop for:
- Early stopping
- Model checkpointing
- TensorBoard logging
- Progress tracking
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class Callback(ABC):
    """
    Base callback class.

    Callbacks are notified of training events and can take
    actions like saving checkpoints or stopping training.
    """

    def __init__(self):
        self.trainer = None

    def set_trainer(self, trainer) -> None:
        """Set the trainer reference."""
        self.trainer = trainer

    def on_train_begin(self, logs: Dict) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, logs: Dict) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, logs: Dict) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, logs: Dict) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, logs: Dict) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, logs: Dict) -> None:
        """Called at the end of each batch."""
        pass

    def on_train_error(self, logs: Dict) -> None:
        """Called when a training error occurs."""
        pass


class CallbackList:
    """Container for multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def add(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def remove(self, callback: Callback) -> None:
        """Remove a callback."""
        self.callbacks.remove(callback)

    def set_trainer(self, trainer) -> None:
        """Set trainer for all callbacks."""
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def notify(self, event: str, logs: Dict) -> None:
        """Notify all callbacks of an event."""
        method_name = f"on_{event}"
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(logs)


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback.

    Stops training when a monitored metric stops improving.
    """

    def __init__(
        self,
        monitor: str = "val_accuracy",
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement
            min_delta: Minimum change to qualify as improvement
            mode: "max" or "min" for metric direction
            restore_best_weights: Whether to restore best weights
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_weights = None
        self.counter = 0
        self.should_stop = False

    def on_train_begin(self, logs: Dict) -> None:
        """Reset state at training start."""
        self.best_value = float("-inf") if self.mode == "max" else float("inf")
        self.best_weights = None
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, logs: Dict) -> None:
        """Check for improvement at epoch end."""
        current = logs.get(self.monitor, logs.get("val_accuracy", 0))

        if self._is_improvement(current):
            self.best_value = current
            self.counter = 0

            if self.restore_best_weights and self.trainer:
                # Save current weights
                self._save_weights()
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\nEarly stopping triggered after {self.counter} epochs without improvement")

                if self.restore_best_weights:
                    self._restore_weights()

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == "max":
            return current > self.best_value + self.min_delta
        else:
            return current < self.best_value - self.min_delta

    def _save_weights(self) -> None:
        """Save current weights."""
        if self.trainer and hasattr(self.trainer, "model"):
            try:
                import torch

                self.best_weights = {
                    k: v.clone() for k, v in self.trainer.model._model.state_dict().items()
                }
            except Exception:
                pass

    def _restore_weights(self) -> None:
        """Restore best weights."""
        if self.best_weights and self.trainer:
            try:
                self.trainer.model._model.load_state_dict(self.best_weights)
                print("Restored best weights")
            except Exception:
                pass


class CheckpointCallback(Callback):
    """
    Model checkpointing callback.

    Saves model weights during training.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        save_best_only: bool = True,
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_frequency: int = 1,
        filename_prefix: str = "model",
    ):
        """
        Initialize checkpointing.

        Args:
            save_dir: Directory to save checkpoints
            save_best_only: Only save when metric improves
            monitor: Metric to monitor
            mode: "max" or "min" for metric direction
            save_frequency: Save every N epochs
            filename_prefix: Prefix for checkpoint files
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_frequency = save_frequency
        self.filename_prefix = filename_prefix

        self.best_value = float("-inf") if mode == "max" else float("inf")

    def on_train_begin(self, logs: Dict) -> None:
        """Create save directory."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_value = float("-inf") if self.mode == "max" else float("inf")

    def on_epoch_end(self, logs: Dict) -> None:
        """Save checkpoint at epoch end."""
        epoch = logs.get("epoch", 0)
        current = logs.get(self.monitor, logs.get("val_accuracy", 0))

        # Check if should save
        if self.save_best_only:
            is_improvement = (
                current > self.best_value if self.mode == "max" else current < self.best_value
            )

            if is_improvement:
                self.best_value = current
                self._save_checkpoint(epoch, current, is_best=True)
        else:
            if (epoch + 1) % self.save_frequency == 0:
                self._save_checkpoint(epoch, current, is_best=False)

    def _save_checkpoint(
        self,
        epoch: int,
        metric_value: float,
        is_best: bool,
    ) -> None:
        """Save model checkpoint."""
        if not self.trainer or not hasattr(self.trainer, "model"):
            return

        try:
            import torch

            model = self.trainer.model._model

            if is_best:
                filename = f"{self.filename_prefix}_best.pt"
            else:
                filename = f"{self.filename_prefix}_epoch{epoch:03d}.pt"

            filepath = self.save_dir / filename

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    f"{self.monitor}": metric_value,
                },
                filepath,
            )

            print(f"Saved checkpoint: {filepath}")

        except Exception as e:
            print(f"Failed to save checkpoint: {e}")


class TensorBoardCallback(Callback):
    """
    TensorBoard logging callback.

    Logs training metrics to TensorBoard.
    """

    def __init__(
        self,
        log_dir: str = "runs/tensorboard",
        log_frequency: int = 1,
        log_images: bool = False,
    ):
        """
        Initialize TensorBoard logging.

        Args:
            log_dir: TensorBoard log directory
            log_frequency: Log every N batches
            log_images: Whether to log sample images
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_frequency = log_frequency
        self.log_images = log_images
        self.writer = None
        self._global_step = 0

    def on_train_begin(self, logs: Dict) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            # Create unique run directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.log_dir / f"run_{timestamp}"
            self.writer = SummaryWriter(str(run_dir))
            print(f"TensorBoard logging to: {run_dir}")

        except ImportError:
            print("TensorBoard not available")
            self.writer = None

    def on_train_end(self, logs: Dict) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()

    def on_epoch_end(self, logs: Dict) -> None:
        """Log epoch metrics."""
        if not self.writer:
            return

        epoch = logs.get("epoch", 0)

        # Log scalar metrics
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"epoch/{key}", value, epoch)

    def on_batch_end(self, logs: Dict) -> None:
        """Log batch metrics."""
        if not self.writer:
            return

        self._global_step += 1

        if self._global_step % self.log_frequency == 0:
            loss = logs.get("loss", 0)
            self.writer.add_scalar("train/batch_loss", loss, self._global_step)


class ProgressCallback(Callback):
    """
    Progress display callback.

    Shows training progress with tqdm or simple prints.
    """

    def __init__(self, use_tqdm: bool = True):
        """
        Initialize progress callback.

        Args:
            use_tqdm: Whether to use tqdm progress bars
        """
        super().__init__()
        self.use_tqdm = use_tqdm
        self.pbar = None
        self.epoch_pbar = None

    def on_train_begin(self, logs: Dict) -> None:
        """Initialize progress bars."""
        epochs = logs.get("epochs", 0)

        if self.use_tqdm:
            try:
                from tqdm import tqdm

                self.epoch_pbar = tqdm(
                    total=epochs,
                    desc="Training",
                    unit="epoch",
                )
            except ImportError:
                self.use_tqdm = False
        else:
            print(f"Starting training for {epochs} epochs")

    def on_train_end(self, logs: Dict) -> None:
        """Close progress bars."""
        if self.epoch_pbar:
            self.epoch_pbar.close()

        result = logs.get("result")
        if result:
            print(f"\nTraining complete!")
            print(f"Best accuracy: {result.best_accuracy:.2%}")
            print(f"Best epoch: {result.best_epoch}")

    def on_epoch_end(self, logs: Dict) -> None:
        """Update progress at epoch end."""
        epoch = logs.get("epoch", 0)
        train_loss = logs.get("train_loss", 0)
        val_accuracy = logs.get("val_accuracy", 0)

        if self.epoch_pbar:
            self.epoch_pbar.update(1)
            self.epoch_pbar.set_postfix(
                {
                    "loss": f"{train_loss:.4f}",
                    "acc": f"{val_accuracy:.2%}",
                }
            )
        else:
            print(f"Epoch {epoch + 1}: " f"loss={train_loss:.4f}, " f"acc={val_accuracy:.2%}")


def create_default_callbacks(
    save_dir: str = "runs/train",
    patience: int = 10,
    tensorboard: bool = True,
) -> List[Callback]:
    """
    Create default set of callbacks.

    Args:
        save_dir: Directory for saving outputs
        patience: Early stopping patience
        tensorboard: Whether to enable TensorBoard

    Returns:
        List of configured callbacks
    """
    callbacks = [
        EarlyStoppingCallback(patience=patience),
        CheckpointCallback(save_dir=save_dir),
        ProgressCallback(),
    ]

    if tensorboard:
        callbacks.append(TensorBoardCallback(log_dir=f"{save_dir}/tensorboard"))

    return callbacks
