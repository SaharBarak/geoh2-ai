"""
Validator - Model Validation and Metrics

Computes validation metrics for trained models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class ValidationResult:
    """Result from model validation."""

    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1: Dict[str, float]
    confusion_matrix: np.ndarray
    per_class_accuracy: Dict[str, float]
    support: Dict[str, int]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_accuracy": self.per_class_accuracy,
            "support": self.support,
        }

    @property
    def macro_precision(self) -> float:
        """Macro-averaged precision."""
        values = list(self.precision.values())
        return sum(values) / len(values) if values else 0.0

    @property
    def macro_recall(self) -> float:
        """Macro-averaged recall."""
        values = list(self.recall.values())
        return sum(values) / len(values) if values else 0.0

    @property
    def macro_f1(self) -> float:
        """Macro-averaged F1 score."""
        values = list(self.f1.values())
        return sum(values) / len(values) if values else 0.0


class Validator:
    """
    Validator for evaluating model performance.

    Computes classification metrics including accuracy,
    precision, recall, F1, and confusion matrix.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize validator.

        Args:
            class_names: List of class names
        """
        self.class_names = class_names or []

    def evaluate(
        self,
        predictions: List,
        ground_truth: List[Union[int, str]],
    ) -> ValidationResult:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: List of PredictionResult objects or predicted labels
            ground_truth: List of ground truth labels (int or str)

        Returns:
            ValidationResult with computed metrics
        """
        # Extract predicted classes
        pred_classes = []
        for p in predictions:
            if hasattr(p, "class_id"):
                pred_classes.append(p.class_id)
            elif hasattr(p, "class_name"):
                pred_classes.append(self._name_to_id(p.class_name))
            else:
                pred_classes.append(int(p))

        # Convert ground truth to ids
        true_classes = []
        for gt in ground_truth:
            if isinstance(gt, str):
                true_classes.append(self._name_to_id(gt))
            else:
                true_classes.append(int(gt))

        pred_classes = np.array(pred_classes)
        true_classes = np.array(true_classes)

        # Compute metrics
        accuracy = self._compute_accuracy(pred_classes, true_classes)
        confusion_matrix = self._compute_confusion_matrix(pred_classes, true_classes)
        precision = self._compute_precision(confusion_matrix)
        recall = self._compute_recall(confusion_matrix)
        f1 = self._compute_f1(precision, recall)
        per_class_accuracy = self._compute_per_class_accuracy(confusion_matrix)
        support = self._compute_support(true_classes)

        return ValidationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=confusion_matrix,
            per_class_accuracy=per_class_accuracy,
            support=support,
        )

    def evaluate_model(
        self,
        model,
        data_loader,
    ) -> ValidationResult:
        """
        Evaluate a model on a data loader.

        Args:
            model: Model with predict method
            data_loader: Data loader yielding (images, labels)

        Returns:
            ValidationResult
        """
        predictions = []
        ground_truth = []

        for images, labels in data_loader:
            # Get predictions
            batch_preds = model.predict_batch(images)
            predictions.extend(batch_preds)

            # Collect ground truth
            if hasattr(labels, "tolist"):
                ground_truth.extend(labels.tolist())
            else:
                ground_truth.extend(list(labels))

        return self.evaluate(predictions, ground_truth)

    def _name_to_id(self, name: str) -> int:
        """Convert class name to id."""
        if name in self.class_names:
            return self.class_names.index(name)
        return -1

    def _id_to_name(self, class_id: int) -> str:
        """Convert class id to name."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def _compute_accuracy(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """Compute overall accuracy."""
        correct = np.sum(predictions == ground_truth)
        total = len(ground_truth)
        return correct / total if total > 0 else 0.0

    def _compute_confusion_matrix(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> np.ndarray:
        """Compute confusion matrix."""
        num_classes = max(
            len(self.class_names), int(max(predictions.max(), ground_truth.max())) + 1
        )

        cm = np.zeros((num_classes, num_classes), dtype=np.int64)

        for pred, true in zip(predictions, ground_truth):
            if 0 <= pred < num_classes and 0 <= true < num_classes:
                cm[true, pred] += 1

        return cm

    def _compute_precision(
        self,
        confusion_matrix: np.ndarray,
    ) -> Dict[str, float]:
        """Compute per-class precision."""
        precision = {}
        num_classes = confusion_matrix.shape[0]

        for i in range(num_classes):
            col_sum = confusion_matrix[:, i].sum()
            if col_sum > 0:
                precision[self._id_to_name(i)] = confusion_matrix[i, i] / col_sum
            else:
                precision[self._id_to_name(i)] = 0.0

        return precision

    def _compute_recall(
        self,
        confusion_matrix: np.ndarray,
    ) -> Dict[str, float]:
        """Compute per-class recall."""
        recall = {}
        num_classes = confusion_matrix.shape[0]

        for i in range(num_classes):
            row_sum = confusion_matrix[i, :].sum()
            if row_sum > 0:
                recall[self._id_to_name(i)] = confusion_matrix[i, i] / row_sum
            else:
                recall[self._id_to_name(i)] = 0.0

        return recall

    def _compute_f1(
        self,
        precision: Dict[str, float],
        recall: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute per-class F1 score."""
        f1 = {}

        for class_name in precision:
            p = precision[class_name]
            r = recall.get(class_name, 0.0)

            if p + r > 0:
                f1[class_name] = 2 * p * r / (p + r)
            else:
                f1[class_name] = 0.0

        return f1

    def _compute_per_class_accuracy(
        self,
        confusion_matrix: np.ndarray,
    ) -> Dict[str, float]:
        """Compute per-class accuracy."""
        accuracy = {}
        num_classes = confusion_matrix.shape[0]

        for i in range(num_classes):
            row_sum = confusion_matrix[i, :].sum()
            if row_sum > 0:
                accuracy[self._id_to_name(i)] = confusion_matrix[i, i] / row_sum
            else:
                accuracy[self._id_to_name(i)] = 0.0

        return accuracy

    def _compute_support(
        self,
        ground_truth: np.ndarray,
    ) -> Dict[str, int]:
        """Compute number of samples per class."""
        support = {}
        unique, counts = np.unique(ground_truth, return_counts=True)

        for class_id, count in zip(unique, counts):
            support[self._id_to_name(class_id)] = int(count)

        return support

    def classification_report(
        self,
        result: ValidationResult,
    ) -> str:
        """
        Generate a classification report string.

        Args:
            result: ValidationResult object

        Returns:
            Formatted classification report
        """
        lines = []
        lines.append("Classification Report")
        lines.append("=" * 60)
        lines.append("")

        # Header
        lines.append(
            f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
        )
        lines.append("-" * 60)

        # Per-class metrics
        for class_name in result.precision:
            p = result.precision.get(class_name, 0)
            r = result.recall.get(class_name, 0)
            f = result.f1.get(class_name, 0)
            s = result.support.get(class_name, 0)
            lines.append(f"{class_name:<20} {p:>10.3f} {r:>10.3f} {f:>10.3f} {s:>10}")

        lines.append("-" * 60)

        # Averages
        lines.append(
            f"{'Macro Avg':<20} {result.macro_precision:>10.3f} {result.macro_recall:>10.3f} {result.macro_f1:>10.3f}"
        )
        lines.append("")
        lines.append(f"Overall Accuracy: {result.accuracy:.3f}")

        return "\n".join(lines)
