"""
DefectLens AI - Inference Engine.

Provides a :class:`DefectPredictor` that loads a trained checkpoint and
runs efficient inference on single images or batches.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Validate and normalise a raw image array.

    Ensures the image is a 3-channel uint8 numpy array in HWC format.

    Args:
        image: Input image array.

    Returns:
        Validated image.

    Raises:
        ValueError: If the image is invalid.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}.")

    if image.size == 0:
        raise ValueError("Image is empty (zero pixels).")

    # Convert grayscale → BGR → RGB-like 3-channel
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(
                f"Unsupported number of channels: {image.shape[2]}. "
                "Expected 1, 3, or 4."
            )
    else:
        raise ValueError(
            f"Unsupported image dimensions: {image.ndim}. "
            "Expected 2 (HW) or 3 (HWC)."
        )

    return image


def load_image(source: Union[str, Path, np.ndarray], image_size: int = 256) -> np.ndarray:
    """Load and resize an image from a path or numpy array.

    Args:
        source: File path or numpy array.
        image_size: Target spatial dimension (square resize).

    Returns:
        Validated RGB uint8 image of shape ``(image_size, image_size, 3)``.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the image cannot be decoded.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"Failed to decode image: {path}")
        image = _validate_image(raw)
    elif isinstance(source, np.ndarray):
        image = _validate_image(source)
    else:
        raise TypeError(
            f"source must be a file path or numpy array, got {type(source).__name__}."
        )

    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return image


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------


class PredictionResult:
    """Container for a single inference result.

    Attributes:
        label: Predicted class index (0 = normal, 1 = defective).
        confidence: Softmax probability for the predicted class.
        probabilities: Full probability vector over all classes.
        anomaly_map: Optional spatial anomaly heatmap (H×W float32).
        latency_ms: Inference latency in milliseconds.
    """

    __slots__ = ("label", "confidence", "probabilities", "anomaly_map", "latency_ms")

    def __init__(
        self,
        label: int,
        confidence: float,
        probabilities: np.ndarray,
        anomaly_map: Optional[np.ndarray] = None,
        latency_ms: float = 0.0,
    ) -> None:
        self.label = label
        self.confidence = confidence
        self.probabilities = probabilities
        self.anomaly_map = anomaly_map
        self.latency_ms = latency_ms

    @property
    def label_name(self) -> str:
        return "defective" if self.label == 1 else "normal"

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "label": self.label,
            "label_name": self.label_name,
            "confidence": round(self.confidence, 4),
            "probabilities": self.probabilities.tolist(),
            "latency_ms": round(self.latency_ms, 2),
            "has_anomaly_map": self.anomaly_map is not None,
        }
        return result

    def __repr__(self) -> str:
        return (
            f"PredictionResult(label={self.label_name!r}, "
            f"conf={self.confidence:.3f}, "
            f"latency={self.latency_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class DefectPredictor:
    """Inference engine for DefectLens classification models.

    Loads a model from a checkpoint and provides methods for running
    inference on individual images or batches.

    Args:
        checkpoint_path: Path to a saved ``best_model.pth`` checkpoint.
        model: Pre-constructed PyTorch model whose state dict will be
            loaded from the checkpoint.  If ``None`` the checkpoint must
            contain the model architecture (not just the state dict).
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        image_size: Expected spatial input size (square).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        model: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "auto",
        image_size: int = 256,
        threshold: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.threshold = threshold

        # Device
        if isinstance(device, torch.device):
            self.device = device
        elif device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info("Inference device: %s", self.device)

        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if model is None:
            from defectlens.models.classifier import DefectClassifier
            model = DefectClassifier(mode="finetune", pretrained=False)

        # Load state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        elif hasattr(checkpoint, "state_dict"):
            model.load_state_dict(checkpoint.state_dict())

        self.model = model.to(self.device)
        self.model.eval()

        # Preprocessing pipeline (must match training transforms)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info(
            "Model loaded from %s (epoch %s, val_loss %.4f)",
            checkpoint_path,
            checkpoint.get("epoch", "?"),
            checkpoint.get("val_loss", float("nan")),
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert a validated RGB uint8 image to a model-ready tensor.

        Args:
            image: RGB image of shape ``(H, W, 3)``, uint8.

        Returns:
            Tensor of shape ``(1, 3, H, W)``.
        """
        tensor = self.transform(image)  # (3, H, W)
        return tensor.unsqueeze(0)  # (1, 3, H, W)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> PredictionResult:
        """Run inference on a single image.

        Args:
            source: Image file path or numpy array.

        Returns:
            :class:`PredictionResult` with label, confidence, and latency.
        """
        image = load_image(source, self.image_size)
        tensor = self._preprocess(image).to(self.device)

        start = time.perf_counter()
        output = self.model(tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if isinstance(output, dict):
            if "probs" in output:
                probs = output["probs"].cpu().numpy().squeeze()
            else:
                probs = F.softmax(output["logits"], dim=1).cpu().numpy().squeeze()
        else:
            probs = F.softmax(output, dim=1).cpu().numpy().squeeze()

        label = int(np.argmax(probs))
        confidence = float(probs[label])

        return PredictionResult(
            label=label,
            confidence=confidence,
            probabilities=probs,
            latency_ms=elapsed_ms,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        sources: Sequence[Union[str, Path, np.ndarray]],
    ) -> List[PredictionResult]:
        """Run inference on a batch of images.

        Args:
            sources: Sequence of image paths or numpy arrays.

        Returns:
            List of :class:`PredictionResult`, one per input image.
        """
        if not sources:
            return []

        # Load and preprocess all images
        tensors: List[torch.Tensor] = []
        for src in sources:
            img = load_image(src, self.image_size)
            tensors.append(self._preprocess(img).squeeze(0))  # (3, H, W)

        batch = torch.stack(tensors).to(self.device)  # (B, 3, H, W)

        start = time.perf_counter()
        outputs = self.model(batch)
        total_ms = (time.perf_counter() - start) * 1000.0
        per_image_ms = total_ms / len(sources)

        if isinstance(outputs, dict):
            if "probs" in outputs:
                probs = outputs["probs"].cpu().numpy()
            else:
                probs = F.softmax(outputs["logits"], dim=1).cpu().numpy()
        else:
            probs = F.softmax(outputs, dim=1).cpu().numpy()  # (B, C)

        labels = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(labels)), labels]

        results: List[PredictionResult] = []
        for i in range(len(sources)):
            results.append(
                PredictionResult(
                    label=int(labels[i]),
                    confidence=float(confidences[i]),
                    probabilities=probs[i],
                    latency_ms=per_image_ms,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    @torch.no_grad()
    def benchmark(
        self,
        source: Optional[Union[str, Path, np.ndarray]] = None,
        num_iterations: int = 100,
        n_runs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Measure inference latency with a dummy input or provided image.

        Args:
            source: Optional image file path or numpy array.
            num_iterations: Number of forward passes to average over.
            n_runs: Alias for num_iterations.

        Returns:
            Dictionary with ``mean_latency_ms``, ``std_latency_ms``,
            ``min_latency_ms``, ``max_latency_ms``, and ``fps``.
        """
        iterations = n_runs if n_runs is not None else num_iterations
        if source is not None:
            image = load_image(source, self.image_size)
            tensor = self._preprocess(image).to(self.device)
        else:
            tensor = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)

        # Warmup
        for _ in range(min(5, iterations)):
            self.model(tensor)

        timings: List[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.model(tensor)
            timings.append((time.perf_counter() - start) * 1000.0)

        mean_ms = float(np.mean(timings))
        std_ms = float(np.std(timings))
        min_ms = float(np.min(timings))
        max_ms = float(np.max(timings))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        return {
            "mean_latency_ms": round(mean_ms, 2),
            "std_latency_ms": round(std_ms, 2),
            "min_latency_ms": round(min_ms, 2),
            "max_latency_ms": round(max_ms, 2),
            "fps": round(fps, 1),
            "num_iterations": iterations,
            "device": str(self.device),
        }
