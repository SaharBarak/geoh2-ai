"""
H2 Seep Detection - Training Module

Provides training infrastructure including:
- Trainer: Main training orchestrator
- Validator: Model validation and metrics
- Callbacks: Training hooks (early stopping, checkpointing, etc.)
"""

from .callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    ProgressCallback,
    TensorBoardCallback,
)
from .trainer import Trainer, TrainingConfig
from .validator import ValidationResult, Validator

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
