"""
DefectLens AI - Evaluation Metrics.

Provides functions for computing binary classification and pixel-level
segmentation metrics for defect detection tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute binary classification metrics.

    Args:
        y_true: Ground-truth binary labels (0 or 1), shape ``(N,)``.
        y_pred: Predicted binary labels (0 or 1), shape ``(N,)``.
        y_scores: Predicted probabilities / confidence scores for the
            positive class, shape ``(N,)``.  Required for AUROC.
        threshold: Decision threshold (used only for documentation;
            ``y_pred`` should already be thresholded).

    Returns:
        Dictionary containing accuracy, precision, recall, F1, and
        optionally AUROC.

    Raises:
        ValueError: If input arrays have mismatched lengths.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)} elements."
        )

    if len(y_true) == 0:
        raise ValueError("Input arrays must not be empty.")

    results: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "num_samples": int(len(y_true)),
    }

    if y_scores is not None:
        y_scores = np.asarray(y_scores).ravel()
        if len(y_scores) != len(y_true):
            raise ValueError(
                f"Length mismatch: y_scores has {len(y_scores)} elements, "
                f"expected {len(y_true)}."
            )
        # AUROC requires both classes to be present
        unique_labels = np.unique(y_true)
        if len(unique_labels) >= 2:
            results["auroc"] = float(roc_auc_score(y_true, y_scores))
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            results["roc_fpr"] = fpr.tolist()
            results["roc_tpr"] = tpr.tolist()
        else:
            results["auroc"] = None
            logger.warning(
                "AUROC is undefined when only one class is present in y_true."
            )

    return results


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute the confusion matrix and derived counts.

    Args:
        y_true: Ground-truth labels, shape ``(N,)``.
        y_pred: Predicted labels, shape ``(N,)``.
        labels: Ordered list of label indices.

    Returns:
        Dictionary with ``matrix`` (list of lists), ``TP``, ``TN``,
        ``FP``, ``FN`` counts (for binary case).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    result: Dict[str, Any] = {"matrix": cm.tolist()}

    # Binary case – extract scalar counts
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        result.update({"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)})

    return result


# ---------------------------------------------------------------------------
# Pixel-level segmentation metrics
# ---------------------------------------------------------------------------


def _validate_masks(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> None:
    """Validate that two masks are compatible shapes.

    Raises:
        ValueError: On shape mismatch or empty arrays.
    """
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: gt_mask {gt_mask.shape} vs "
            f"pred_mask {pred_mask.shape}."
        )
    if gt_mask.size == 0:
        raise ValueError("Masks must not be empty.")


def compute_iou(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Intersection-over-Union (IoU / Jaccard index).

    Args:
        gt_mask: Ground-truth binary mask.
        pred_mask: Predicted mask (probabilities or binary).
        threshold: Binarisation threshold for ``pred_mask``.

    Returns:
        IoU score in [0, 1].
    """
    gt = (np.asarray(gt_mask) > threshold).astype(bool)
    pred = (np.asarray(pred_mask) > threshold).astype(bool)
    _validate_masks(gt, pred)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return 1.0  # both empty → perfect match
    return float(intersection / union)


def compute_dice(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute the Dice coefficient (F1 at pixel level).

    Args:
        gt_mask: Ground-truth binary mask.
        pred_mask: Predicted mask (probabilities or binary).
        threshold: Binarisation threshold for ``pred_mask``.

    Returns:
        Dice score in [0, 1].
    """
    gt = (np.asarray(gt_mask) > threshold).astype(bool)
    pred = (np.asarray(pred_mask) > threshold).astype(bool)
    _validate_masks(gt, pred)

    intersection = np.logical_and(gt, pred).sum()
    total = gt.sum() + pred.sum()

    if total == 0:
        return 1.0  # both empty
    return float(2.0 * intersection / total)


def compute_pixel_auroc(
    gt_mask: np.ndarray,
    anomaly_map: np.ndarray,
) -> Optional[float]:
    """Compute pixel-level AUROC from a continuous anomaly map.

    Args:
        gt_mask: Ground-truth binary mask (0/1), any shape.
        anomaly_map: Continuous predicted anomaly scores, same shape as
            ``gt_mask``.

    Returns:
        Pixel AUROC score, or ``None`` if only one class is present.
    """
    gt = np.asarray(gt_mask).ravel()
    scores = np.asarray(anomaly_map).ravel()
    _validate_masks(
        np.asarray(gt_mask),
        np.asarray(anomaly_map),
    )

    unique = np.unique(gt)
    if len(unique) < 2:
        logger.warning(
            "Pixel AUROC is undefined when only one class is present in the mask."
        )
        return None

    gt_binary = (gt > 0).astype(int)
    return float(roc_auc_score(gt_binary, scores))


def compute_segmentation_metrics(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    anomaly_map: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute a full set of pixel-level segmentation metrics.

    Args:
        gt_mask: Ground-truth binary mask.
        pred_mask: Predicted binary or probability mask.
        anomaly_map: Optional continuous anomaly map for pixel AUROC.
        threshold: Binarisation threshold.

    Returns:
        Dictionary with IoU, Dice, pixel accuracy, and optionally pixel AUROC.
    """
    gt = (np.asarray(gt_mask) > threshold).astype(bool).ravel()
    pred = (np.asarray(pred_mask) > threshold).astype(bool).ravel()

    results: Dict[str, Any] = {
        "iou": compute_iou(gt_mask, pred_mask, threshold),
        "dice": compute_dice(gt_mask, pred_mask, threshold),
        "pixel_accuracy": float(np.mean(gt == pred)),
        "threshold": threshold,
    }

    if anomaly_map is not None:
        results["pixel_auroc"] = compute_pixel_auroc(gt_mask, anomaly_map)

    return results
