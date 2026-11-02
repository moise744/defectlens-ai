"""
DefectLens AI - Error Analysis Utilities.

Provides tools for identifying misclassified samples, grouping errors by
defect type, computing per-category metrics, and finding the hardest examples.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class PredictionRecord:
    """Lightweight container for a single prediction result.

    Args:
        image_id: Unique identifier (e.g. filename or index).
        y_true: Ground-truth label (0 = normal, 1 = defective).
        y_pred: Predicted label after thresholding.
        confidence: Model confidence / probability for predicted class.
        loss: Per-sample loss value (optional).
        category: Defect category name (optional).
    """

    __slots__ = ("image_id", "y_true", "y_pred", "confidence", "loss", "category")

    def __init__(
        self,
        image_id: Union[str, int],
        y_true: int,
        y_pred: int,
        confidence: float,
        loss: Optional[float] = None,
        category: Optional[str] = None,
    ) -> None:
        self.image_id = image_id
        self.y_true = int(y_true)
        self.y_pred = int(y_pred)
        self.confidence = float(confidence)
        self.loss = float(loss) if loss is not None else None
        self.category = category

    @property
    def is_correct(self) -> bool:
        return self.y_true == self.y_pred

    @property
    def is_false_positive(self) -> bool:
        """Predicted defective but actually normal."""
        return self.y_true == 0 and self.y_pred == 1

    @property
    def is_false_negative(self) -> bool:
        """Predicted normal but actually defective."""
        return self.y_true == 1 and self.y_pred == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "confidence": self.confidence,
            "loss": self.loss,
            "category": self.category,
            "correct": self.is_correct,
        }


# ---------------------------------------------------------------------------
# Error identification
# ---------------------------------------------------------------------------


def find_false_positives(
    records: Sequence[PredictionRecord],
) -> List[PredictionRecord]:
    """Return all false-positive predictions (predicted defective, actually normal).

    Args:
        records: Sequence of :class:`PredictionRecord`.

    Returns:
        List of false-positive records sorted by confidence (descending).
    """
    fps = [r for r in records if r.is_false_positive]
    fps.sort(key=lambda r: r.confidence, reverse=True)
    return fps


def find_false_negatives(
    records: Sequence[PredictionRecord],
) -> List[PredictionRecord]:
    """Return all false-negative predictions (predicted normal, actually defective).

    Args:
        records: Sequence of :class:`PredictionRecord`.

    Returns:
        List of false-negative records sorted by confidence (descending).
    """
    fns = [r for r in records if r.is_false_negative]
    fns.sort(key=lambda r: r.confidence, reverse=True)
    return fns


# ---------------------------------------------------------------------------
# Per-category metrics
# ---------------------------------------------------------------------------


def group_by_category(
    records: Sequence[PredictionRecord],
) -> Dict[str, List[PredictionRecord]]:
    """Group prediction records by their ``category`` field.

    Args:
        records: Sequence of :class:`PredictionRecord`.

    Returns:
        Dictionary mapping category names to lists of records.  Records
        without a category are grouped under ``"unknown"``.
    """
    groups: Dict[str, List[PredictionRecord]] = {}
    for r in records:
        key = r.category if r.category is not None else "unknown"
        groups.setdefault(key, []).append(r)
    return groups


def compute_per_category_metrics(
    records: Sequence[PredictionRecord],
) -> Dict[str, Dict[str, Any]]:
    """Compute accuracy, precision, recall, and F1 per category.

    Args:
        records: Sequence of :class:`PredictionRecord`.

    Returns:
        Dictionary mapping each category to its metric dictionary.
    """
    groups = group_by_category(records)
    results: Dict[str, Dict[str, Any]] = {}

    for category, group in groups.items():
        y_true = np.array([r.y_true for r in group])
        y_pred = np.array([r.y_pred for r in group])

        results[category] = {
            "num_samples": len(group),
            "num_errors": int(np.sum(y_true != y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "false_positives": sum(1 for r in group if r.is_false_positive),
            "false_negatives": sum(1 for r in group if r.is_false_negative),
        }

    return results


# ---------------------------------------------------------------------------
# Hardest examples
# ---------------------------------------------------------------------------


def find_hardest_by_loss(
    records: Sequence[PredictionRecord],
    top_k: int = 20,
) -> List[PredictionRecord]:
    """Find the samples with the highest per-sample loss.

    Args:
        records: Sequence of :class:`PredictionRecord` (must have ``loss`` set).
        top_k: Number of examples to return.

    Returns:
        List of up to ``top_k`` records sorted by loss (descending).
    """
    with_loss = [r for r in records if r.loss is not None]
    with_loss.sort(key=lambda r: r.loss, reverse=True)  # type: ignore[arg-type]
    return with_loss[:top_k]


def find_hardest_by_confidence(
    records: Sequence[PredictionRecord],
    top_k: int = 20,
) -> List[PredictionRecord]:
    """Find the most confidently wrong predictions.

    These are misclassified samples where the model was most confident
    in its (incorrect) prediction.

    Args:
        records: Sequence of :class:`PredictionRecord`.
        top_k: Number of examples to return.

    Returns:
        List of up to ``top_k`` wrong records sorted by confidence (descending).
    """
    wrong = [r for r in records if not r.is_correct]
    wrong.sort(key=lambda r: r.confidence, reverse=True)
    return wrong[:top_k]


# ---------------------------------------------------------------------------
# Full error analysis report
# ---------------------------------------------------------------------------


def generate_error_report(
    records: Sequence[PredictionRecord],
    top_k: int = 20,
) -> Dict[str, Any]:
    """Generate a comprehensive error analysis report.

    Args:
        records: Full set of prediction records.
        top_k: Number of hardest examples to include.

    Returns:
        Dictionary containing summary statistics, per-category metrics,
        false-positive / false-negative lists, and hardest examples.
    """
    y_true = np.array([r.y_true for r in records])
    y_pred = np.array([r.y_pred for r in records])

    fps = find_false_positives(records)
    fns = find_false_negatives(records)
    hardest_loss = find_hardest_by_loss(records, top_k)
    hardest_conf = find_hardest_by_confidence(records, top_k)
    per_cat = compute_per_category_metrics(records)

    report: Dict[str, Any] = {
        "summary": {
            "total_samples": len(records),
            "correct": int(np.sum(y_true == y_pred)),
            "incorrect": int(np.sum(y_true != y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "num_false_positives": len(fps),
            "num_false_negatives": len(fns),
        },
        "per_category": per_cat,
        "false_positives": [r.to_dict() for r in fps],
        "false_negatives": [r.to_dict() for r in fns],
        "hardest_by_loss": [r.to_dict() for r in hardest_loss],
        "hardest_by_confidence": [r.to_dict() for r in hardest_conf],
    }

    return report


def save_error_report(
    report: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """Persist an error-analysis report to a JSON file.

    Args:
        report: Report dictionary produced by :func:`generate_error_report`.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Error report saved to %s", path)
