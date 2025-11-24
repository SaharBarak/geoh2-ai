"""
Metrics Computation and Visualization
Evaluates model performance
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_prob: Prediction probabilities (optional, for ROC)
        class_names: List of class names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Per-class metrics
    metrics["precision_micro"] = precision_score(y_true, y_pred, average="micro")
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro")
    metrics["recall_micro"] = recall_score(y_true, y_pred, average="micro")
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro")
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro")
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")

    # Per-class detailed metrics
    if class_names:
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        metrics["per_class"] = report

        # SCD-specific metrics (class 0)
        if "SCD" in class_names:
            scd_metrics = report.get("SCD", {})
            metrics["scd_precision"] = scd_metrics.get("precision", 0.0)
            metrics["scd_recall"] = scd_metrics.get("recall", 0.0)
            metrics["scd_f1"] = scd_metrics.get("f1-score", 0.0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm

    # ROC AUC if probabilities provided
    if y_prob is not None and len(np.unique(y_true)) == 2:
        # Binary classification
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix using seaborn.

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize values
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_class_distribution(
    y: List[int],
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of classes.

    Args:
        y: Class labels
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    unique, counts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(unique)), counts, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
