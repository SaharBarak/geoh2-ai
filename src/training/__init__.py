"""
H2 Seep Detection - Training Module

Provides training infrastructure including:
- Trainer: Main training orchestrator
- Validator: Model validation and metrics
- Callbacks: Training hooks (early stopping, checkpointing, etc.)
"""

from .trainer import Trainer, TrainingConfig
from .validator import Validator, ValidationResult
from .callbacks import (
    Callback,
    CallbackList,
    EarlyStoppingCallback,
    CheckpointCallback,
    TensorBoardCallback,
    ProgressCallback,
)

__all__ = [
    "Trainer",
    "TrainingConfig",
    "Validator",
    "ValidationResult",
    "Callback",
    "CallbackList",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "TensorBoardCallback",
    "ProgressCallback",
]
