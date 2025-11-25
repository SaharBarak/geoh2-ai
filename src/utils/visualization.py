"""
Visualization Utilities

Provides visualization tools for:
- Confusion matrices
- ROC curves
- Confidence heatmaps
- Geographic plots
- Training metrics
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    normalize: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot confusion matrix with labels.

    Args:
        confusion_matrix: 2D array of confusion matrix values
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        normalize: Whether to normalize values
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize if requested
    if normalize:
        cm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cm = confusion_matrix

    # Plot
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = f"{cm[i, j]:.2f}" if normalize else f"{confusion_matrix[i, j]}"
            ax.text(j, i, value,
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=8)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_roc_curves(
    fpr_dict: Dict[str, np.ndarray],
    tpr_dict: Dict[str, np.ndarray],
    auc_dict: Dict[str, float],
    title: str = "ROC Curves",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Plot ROC curves for multiple classes.

    Args:
        fpr_dict: False positive rates per class
        tpr_dict: True positive rates per class
        auc_dict: AUC values per class
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each class
    colors = plt.cm.tab10(np.linspace(0, 1, len(fpr_dict)))

    for (class_name, fpr), color in zip(fpr_dict.items(), colors):
        tpr = tpr_dict[class_name]
        auc = auc_dict.get(class_name, 0)

        ax.plot(
            fpr, tpr, color=color,
            label=f'{class_name} (AUC = {auc:.3f})'
        )

    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
):
    """
    Plot training history metrics.

    Args:
        history: Dictionary of metric lists
        metrics: Metrics to plot (defaults to all)
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = list(history.keys())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in history:
            values = history[metric]
            epochs = range(1, len(values) + 1)

            ax.plot(epochs, values, 'b-', label=metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_confidence_histogram(
    predictions: List,
    class_name: Optional[str] = None,
    bins: int = 20,
    title: str = "Confidence Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot histogram of prediction confidences.

    Args:
        predictions: List of PredictionResult objects
        class_name: Filter by specific class
        bins: Number of histogram bins
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    # Extract confidences
    if class_name:
        confidences = [
            p.confidence for p in predictions
            if p.class_name == class_name
        ]
        title = f"{title} - {class_name}"
    else:
        confidences = [p.confidence for p in predictions]

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(confidences, bins=bins, edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='r', linestyle='--', label='Threshold (0.5)')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_predictions_on_map(
    predictions: List,
    coordinates: List[Tuple[float, float]],
    title: str = "Predictions Map",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
):
    """
    Plot predictions on a geographic map.

    Args:
        predictions: List of PredictionResult objects
        coordinates: List of (lon, lat) tuples
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Separate by class
    scd_coords = []
    scd_conf = []
    other_coords = []
    other_conf = []

    for pred, (lon, lat) in zip(predictions, coordinates):
        if pred.class_name == "SCD":
            scd_coords.append((lon, lat))
            scd_conf.append(pred.confidence)
        else:
            other_coords.append((lon, lat))
            other_conf.append(pred.confidence)

    # Plot other classes
    if other_coords:
        lons, lats = zip(*other_coords)
        ax.scatter(lons, lats, c='gray', s=30, alpha=0.5, label='Non-SCD')

    # Plot SCDs with confidence coloring
    if scd_coords:
        lons, lats = zip(*scd_coords)
        scatter = ax.scatter(
            lons, lats, c=scd_conf, s=50,
            cmap='RdYlGn', vmin=0.5, vmax=1.0,
            edgecolors='black', linewidths=0.5,
            label='SCD'
        )
        plt.colorbar(scatter, ax=ax, label='Confidence')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_class_distribution(
    predictions: List,
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot bar chart of class distribution.

    Args:
        predictions: List of PredictionResult objects
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    # Count classes
    class_counts = Counter(p.class_name for p in predictions)

    fig, ax = plt.subplots(figsize=figsize)

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # Color SCD differently
    colors = ['green' if c == 'SCD' else 'steelblue' for c in classes]

    bars = ax.bar(classes, counts, color=colors, edgecolor='black')

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha='center', va='bottom'
        )

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def visualize_spectral_indices(
    indices: Dict[str, np.ndarray],
    title: str = "Spectral Indices",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize multiple spectral indices.

    Args:
        indices: Dictionary of index arrays
        title: Plot title
        figsize: Figure size (auto if None)
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    n_indices = len(indices)
    cols = min(3, n_indices)
    rows = (n_indices + cols - 1) // cols

    if figsize is None:
        figsize = (4 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for ax, (name, values) in zip(axes, indices.items()):
        # Handle IndexResult objects
        if hasattr(values, 'value'):
            data = values.value
            vmin, vmax = values.valid_range
        else:
            data = values
            vmin, vmax = np.nanmin(data), np.nanmax(data)

        im = ax.imshow(data, cmap='RdYlGn', vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for ax in axes[n_indices:]:
        ax.axis('off')

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def create_prediction_report(
    predictions: List,
    coordinates: List[Tuple[float, float]],
    output_dir: str = "reports",
    report_name: str = "prediction_report",
) -> str:
    """
    Create a comprehensive prediction report with visualizations.

    Args:
        predictions: List of PredictionResult objects
        coordinates: List of (lon, lat) tuples
        output_dir: Output directory
        report_name: Report name

    Returns:
        Path to generated report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_class_distribution(
        predictions,
        save_path=str(output_path / f"{report_name}_class_dist.png")
    )

    plot_confidence_histogram(
        predictions,
        save_path=str(output_path / f"{report_name}_confidence.png")
    )

    plot_predictions_on_map(
        predictions, coordinates,
        save_path=str(output_path / f"{report_name}_map.png")
    )

    # Generate summary text
    from collections import Counter

    class_counts = Counter(p.class_name for p in predictions)
    scd_count = class_counts.get('SCD', 0)
    total = len(predictions)

    high_conf_scds = sum(1 for p in predictions if p.class_name == 'SCD' and p.confidence > 0.7)

    summary = f"""
# Prediction Report: {report_name}

## Summary
- Total structures analyzed: {total}
- Potential H2 seeps (SCDs): {scd_count} ({100*scd_count/total:.1f}%)
- High-confidence SCDs (>70%): {high_conf_scds}

## Class Distribution
"""
    for class_name, count in class_counts.most_common():
        pct = 100 * count / total
        summary += f"- {class_name}: {count} ({pct:.1f}%)\n"

    summary += f"""
## Generated Files
- {report_name}_class_dist.png - Class distribution chart
- {report_name}_confidence.png - Confidence distribution
- {report_name}_map.png - Geographic map of predictions
"""

    # Save summary
    summary_path = output_path / f"{report_name}_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary)

    return str(summary_path)
