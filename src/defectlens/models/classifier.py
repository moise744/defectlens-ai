"""
Binary defect classifier based on ResNet-18.

Classifies input images as **normal** (0) or **defective** (1) using a
pretrained ResNet-18 backbone with a custom classification head.
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class DefectClassifier(nn.Module):
    """Binary (normal / defective) classifier.

    Parameters
    ----------
    mode : ``"finetune"`` | ``"feature_extract"``
        ``"finetune"`` — all parameters are trainable.
        ``"feature_extract"`` — backbone is frozen; only the
        classification head is trained.
    num_classes : int
        Number of output classes (default 2 for binary).
    dropout : float
        Dropout probability before the final FC layer.
    pretrained : bool
        Load ImageNet-pretrained weights for the backbone.
    """

    def __init__(
        self,
        mode: Literal["finetune", "feature_extract"] = "finetune",
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes

        # ── Backbone ─────────────────────────────────────────────────
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features  # 512 for ResNet-18

        # Remove the original FC layer; we replace it below
        self.backbone.fc = nn.Identity()

        # Optionally freeze backbone
        if mode == "feature_extract":
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen — feature-extraction mode.")

        # ── Classification head ──────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self._device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, 3, H, W)``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``logits``  — raw scores, shape ``(B, num_classes)``
            ``probs``   — softmax probabilities, same shape
        """
        features = self.backbone(x)          # (B, 512)
        logits = self.classifier(features)   # (B, num_classes)
        probs = torch.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(predicted_labels, confidence)`` for input batch *x*.

        Parameters
        ----------
        x : torch.Tensor
            Input batch ``(B, 3, H, W)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``labels`` — predicted class indices ``(B,)``
            ``confidence`` — max probability ``(B,)``
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            probs = out["probs"]
            confidence, labels = probs.max(dim=1)
        return labels, confidence

    def set_mode(self, mode: Literal["finetune", "feature_extract"]) -> None:
        """Switch between fine-tuning and feature-extraction modes."""
        self.mode = mode
        requires_grad = mode == "finetune"
        for param in self.backbone.parameters():
            param.requires_grad = requires_grad
        state = "unfrozen" if requires_grad else "frozen"
        logger.info("Backbone %s — %s mode.", state, mode)

    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def model_summary(self) -> str:
        """Return a summary string describing the model."""
        total = self.get_total_params()
        trainable = self.get_trainable_params()
        frozen = total - trainable
        lines = [
            f"{'═' * 50}",
            f"  DefectClassifier Summary",
            f"{'═' * 50}",
            f"  Mode          : {self.mode}",
            f"  Backbone      : ResNet-18",
            f"  Num classes   : {self.num_classes}",
            f"  Total params  : {total:,}",
            f"  Trainable     : {trainable:,}",
            f"  Frozen        : {frozen:,}",
            f"{'═' * 50}",
        ]
        return "\n".join(lines)


def build_classifier(
    mode: Literal["finetune", "feature_extract"] = "finetune",
    num_classes: int = 2,
    dropout: float = 0.3,
    pretrained: bool = True,
    device: str | torch.device = "auto",
) -> DefectClassifier:
    """Factory function to create and move a classifier to the target device.

    Parameters
    ----------
    device : str | torch.device
        ``"auto"`` selects CUDA if available, otherwise CPU.
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = DefectClassifier(
        mode=mode,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )
    model = model.to(device)
    model._device = device
    logger.info("Classifier created on %s.", device)
    return model
