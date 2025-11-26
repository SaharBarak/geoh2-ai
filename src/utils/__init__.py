"""
Utility Module
Common utilities and helpers
"""

from .logger import setup_logger
from .metrics import compute_metrics, plot_confusion_matrix
from .visualization import (
    create_prediction_report,
    plot_class_distribution,
    plot_confidence_histogram,
)
from .visualization import plot_confusion_matrix as plot_cm
from .visualization import (
    plot_predictions_on_map,
    plot_roc_curves,
    plot_training_history,
    visualize_spectral_indices,
)

__all__ = [
    "setup_logger",
    "compute_metrics",
    "plot_confusion_matrix",
    "plot_cm",
    "plot_roc_curves",
    "plot_training_history",
    "plot_confidence_histogram",
    "plot_predictions_on_map",
    "plot_class_distribution",
    "visualize_spectral_indices",
    "create_prediction_report",
]
