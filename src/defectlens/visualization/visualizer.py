"""
DefectLens AI - Visualization Utilities.

Provides functions for drawing bounding boxes, overlaying segmentation masks,
generating anomaly heatmaps, plotting training curves, confusion matrices,
and ROC curves.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Default colour palette (BGR for OpenCV)
_PALETTE_BGR: List[Tuple[int, int, int]] = [
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Union[str, Path]) -> Path:
    """Create parent directories and return a Path object."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image to uint8 if it is floating-point."""
    if image.dtype in (np.float32, np.float64):
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


# ---------------------------------------------------------------------------
# Bounding box drawing
# ---------------------------------------------------------------------------


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: Sequence[Sequence[int]],
    labels: Optional[Sequence[str]] = None,
    scores: Optional[Sequence[float]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes on an image.

    Args:
        image: Input image (BGR uint8).
        boxes: Sequence of ``[x1, y1, x2, y2]`` bounding boxes.
        labels: Optional class names for each box.
        scores: Optional confidence scores for each box.
        color: BGR colour tuple; cycles through a palette if ``None``.
        thickness: Box line thickness.
        font_scale: Font scale for labels.

    Returns:
        Annotated copy of the input image.
    """
    canvas = image.copy()

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        c = color if color is not None else _PALETTE_BGR[idx % len(_PALETTE_BGR)]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), c, thickness)

        # Build label string
        parts: List[str] = []
        if labels is not None and idx < len(labels):
            parts.append(labels[idx])
        if scores is not None and idx < len(scores):
            parts.append(f"{scores[idx]:.2f}")
        text = " ".join(parts)

        if text:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
            cv2.putText(
                canvas, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

    return canvas


# ---------------------------------------------------------------------------
# Segmentation mask overlay
# ---------------------------------------------------------------------------


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a binary segmentation mask on an image with transparency.

    Args:
        image: Input image (BGR uint8), shape ``(H, W, 3)``.
        mask: Binary mask (0/1 or bool), shape ``(H, W)``.
        color: BGR overlay colour.
        alpha: Transparency factor (0 = invisible, 1 = opaque).

    Returns:
        Image with mask overlay.
    """
    canvas = image.copy()
    mask_bool = mask.astype(bool)

    overlay = canvas.copy()
    overlay[mask_bool] = color
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    return canvas


# ---------------------------------------------------------------------------
# Anomaly heatmap
# ---------------------------------------------------------------------------


def create_anomaly_heatmap(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Create a colour-mapped anomaly heatmap overlaid on the input image.

    Args:
        image: Original image (BGR uint8), shape ``(H, W, 3)``.
        anomaly_map: Continuous anomaly scores, shape ``(H, W)``.
            Values are min-max normalised internally.
        alpha: Blending factor for the heatmap.
        colormap: OpenCV colourmap constant.

    Returns:
        Blended heatmap image (BGR uint8).
    """
    # Normalise to 0-255
    a_min, a_max = float(anomaly_map.min()), float(anomaly_map.max())
    if a_max - a_min > 1e-8:
        normed = ((anomaly_map - a_min) / (a_max - a_min) * 255).astype(np.uint8)
    else:
        normed = np.zeros_like(anomaly_map, dtype=np.uint8)

    heatmap = cv2.applyColorMap(normed, colormap)

    # Resize heatmap to match image
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    blended = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    return blended


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------


def side_by_side(
    images: Sequence[np.ndarray],
    titles: Optional[Sequence[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> Figure:
    """Create a side-by-side comparison of multiple images.

    Args:
        images: Sequence of images (BGR uint8 or RGB).
        titles: Optional title for each image.
        save_path: If provided, save the figure to disk.
        figsize: Figure size in inches.

    Returns:
        Matplotlib :class:`Figure`.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for i, (img, ax) in enumerate(zip(images, axes)):
        display = cv2.cvtColor(_to_uint8(img), cv2.COLOR_BGR2RGB)
        ax.imshow(display)
        ax.axis("off")
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        p = _ensure_dir(save_path)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        logger.info("Side-by-side saved to %s", p)

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Prediction grid
# ---------------------------------------------------------------------------


def prediction_grid(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    confidences: Optional[Sequence[float]] = None,
    ncols: int = 4,
    save_path: Optional[Union[str, Path]] = None,
    figsize_per_cell: Tuple[float, float] = (3.0, 3.5),
) -> Figure:
    """Visualise a grid of predictions.

    Args:
        images: Sequence of images (BGR uint8).
        labels: Predicted label string for each image.
        confidences: Optional confidence score for each image.
        ncols: Number of columns in the grid.
        save_path: If provided, save the figure to disk.
        figsize_per_cell: ``(width, height)`` in inches per cell.

    Returns:
        Matplotlib :class:`Figure`.
    """
    n = len(images)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_cell[0], nrows * figsize_per_cell[1]),
    )
    axes_flat = np.asarray(axes).ravel()

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < n:
            display = cv2.cvtColor(_to_uint8(images[i]), cv2.COLOR_BGR2RGB)
            ax.imshow(display)
            title = labels[i]
            if confidences is not None and i < len(confidences):
                title += f"\n({confidences[i]:.2%})"
            ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        p = _ensure_dir(save_path)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        logger.info("Prediction grid saved to %s", p)

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix heatmap
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6),
) -> Figure:
    """Plot a confusion matrix as a heatmap.

    Args:
        matrix: Confusion matrix array of shape ``(C, C)``.
        class_names: Label names for each class.
        save_path: If provided, save the figure to disk.
        title: Plot title.
        cmap: Matplotlib colourmap name.
        figsize: Figure size in inches.

    Returns:
        Matplotlib :class:`Figure`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)

    n_classes = matrix.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    thresh = matrix.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, f"{matrix[i, j]}",
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.tight_layout()

    if save_path is not None:
        p = _ensure_dir(save_path)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", p)

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> Figure:
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc`` (lists of floats per epoch).
        save_path: If provided, save the figure to disk.
        figsize: Figure size in inches.

    Returns:
        Matplotlib :class:`Figure`.
    """
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Loss
    if "train_loss" in history:
        ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    if "train_acc" in history:
        ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    if "val_acc" in history:
        ax2.plot(epochs, history["val_acc"], "r-o", markersize=3, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        p = _ensure_dir(save_path)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        logger.info("Training curves saved to %s", p)

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (7, 6),
) -> Figure:
    """Plot a Receiver Operating Characteristic curve.

    Args:
        fpr: False-positive rates array.
        tpr: True-positive rates array.
        auc_score: Area under the curve (shown in legend).
        save_path: If provided, save the figure to disk.
        title: Plot title.
        figsize: Figure size in inches.

    Returns:
        Matplotlib :class:`Figure`.
    """
    fig, ax = plt.subplots(figsize=figsize)

    label = "ROC"
    if auc_score is not None:
        label += f" (AUC = {auc_score:.4f})"

    ax.plot(fpr, tpr, "b-", linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()

    if save_path is not None:
        p = _ensure_dir(save_path)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved to %s", p)

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Convenience: save any OpenCV image
# ---------------------------------------------------------------------------


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
) -> Path:
    """Save a BGR uint8 image to disk.

    Args:
        image: Image array.
        path: Destination file path.

    Returns:
        Resolved :class:`Path` of saved file.
    """
    p = _ensure_dir(path)
    success = cv2.imwrite(str(p), _to_uint8(image))
    if not success:
        raise IOError(f"Failed to save image to {p}")
    logger.info("Image saved to %s", p)
    return p
