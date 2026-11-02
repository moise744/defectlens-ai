"""
Simplified PatchCore-style anomaly detection.

Extracts patch-level features from a pretrained ResNet backbone,
builds a memory bank of *normal* features, and scores new images
by their distance to the nearest stored feature.

This module is designed to work on **CPU** without issues.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)


class _FeatureExtractor(nn.Module):
    """Extracts intermediate feature maps from a pretrained ResNet.

    Hooks into ``layer2`` and ``layer3`` to capture mid-level features
    with reasonable spatial resolution.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Build a truncated sequential up to layer3
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 64-ch,  H/4
        self.layer2 = backbone.layer2  # 128-ch, H/8
        self.layer3 = backbone.layer3  # 256-ch, H/16

        # Freeze all parameters — we never train the extractor
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return concatenated feature map from layer2 + layer3.

        Both maps are upsampled to the spatial size of layer2, then
        concatenated along the channel dimension → ``(B, 384, h, w)``.
        """
        x = self.layer0(x)
        x = self.layer1(x)
        feat2 = self.layer2(x)   # (B, 128, h, w)
        feat3 = self.layer3(feat2)  # (B, 256, h/2, w/2)

        # Up-sample layer3 to match layer2 spatial dims
        feat3_up = F.interpolate(
            feat3, size=feat2.shape[2:], mode="bilinear", align_corners=False
        )
        combined = torch.cat([feat2, feat3_up], dim=1)  # (B, 384, h, w)
        return combined


class PatchCoreAnomalyDetector:
    """Anomaly detector using a normal-image feature memory bank.

    Workflow
    --------
    1. **Fit** — feed normal training images to build a memory bank.
    2. **Score** — compute per-image anomaly scores and per-pixel
       anomaly maps for new images.

    Parameters
    ----------
    device : str | torch.device
        ``"auto"`` selects CUDA if available, else CPU.
    max_memory_samples : int
        Maximum number of patch features to store.  If the raw bank
        exceeds this, features are randomly sub-sampled (with a fixed
        seed for reproducibility).
    seed : int
        Random seed for sub-sampling.
    """

    def __init__(
        self,
        device: str | torch.device = "auto",
        max_memory_samples: int = 25_000,
        seed: int = 42,
        pretrained: bool = True,
    ) -> None:
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.max_memory_samples = max_memory_samples
        self.seed = seed
        self.extractor = _FeatureExtractor(pretrained=pretrained).to(self.device).eval()
        self.memory_bank: Optional[torch.Tensor] = None  # (N, C)
        self._is_fitted = False

        logger.info("PatchCoreAnomalyDetector initialised on %s.", self.device)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Build the feature memory bank from a dataloader of *normal* images.

        Parameters
        ----------
        dataloader : DataLoader
            Must yield dicts with an ``"image"`` key containing tensors
            of shape ``(B, 3, H, W)``.
        """
        all_features: list[torch.Tensor] = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            feat_map = self.extractor(images)  # (B, C, h, w)
            B, C, h, w = feat_map.shape
            # Reshape to (B*h*w, C) — one vector per spatial patch
            patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C)
            all_features.append(patches.cpu())

        bank = torch.cat(all_features, dim=0)  # (total_patches, C)
        logger.info("Raw memory bank: %d patches, %d dims.", bank.shape[0], bank.shape[1])

        # Sub-sample if too large
        if bank.shape[0] > self.max_memory_samples:
            rng = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(bank.shape[0], generator=rng)[
                : self.max_memory_samples
            ]
            bank = bank[indices]
            logger.info("Sub-sampled memory bank to %d patches.", bank.shape[0])

        self.memory_bank = bank.to(self.device)
        self._is_fitted = True
        logger.info("Memory bank fitted: shape %s.", list(self.memory_bank.shape))

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score(
        self, images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute anomaly scores for a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Batch ``(B, 3, H, W)``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``anomaly_score`` — per-image scalar score ``(B,)``
            ``anomaly_map``   — spatial anomaly map ``(B, 1, H, W)``
                                (up-sampled to input resolution)
        """
        if not self._is_fitted or self.memory_bank is None:
            raise RuntimeError(
                "Detector is not fitted. Call .fit() with normal training data first."
            )

        images = images.to(self.device)
        B, _, H, W = images.shape
        feat_map = self.extractor(images)  # (B, C, h, w)
        _, C, h, w = feat_map.shape

        # Flatten spatial dims → (B*h*w, C)
        patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C)

        # Compute pairwise distances in manageable chunks
        dist_map = self._nearest_distance(patches)  # (B*h*w,)
        dist_map = dist_map.view(B, 1, h, w)

        # Up-sample to original image size
        anomaly_map = F.interpolate(
            dist_map, size=(H, W), mode="bilinear", align_corners=False
        )

        # Per-image score = max patch distance
        anomaly_score = anomaly_map.view(B, -1).max(dim=1).values  # (B,)

        return {
            "anomaly_score": anomaly_score,
            "anomaly_map": anomaly_map,
        }

    def _nearest_distance(self, queries: torch.Tensor) -> torch.Tensor:
        """Return the distance of each query to its nearest memory-bank vector.

        Uses chunked computation to avoid OOM on CPU.
        """
        assert self.memory_bank is not None
        chunk_size = 2048
        distances: list[torch.Tensor] = []

        for i in range(0, queries.shape[0], chunk_size):
            q = queries[i : i + chunk_size]  # (chunk, C)
            # L2 distance via expanded form: ||q - m||^2 = ||q||^2 + ||m||^2 - 2*q·m
            q_sq = (q ** 2).sum(dim=1, keepdim=True)          # (chunk, 1)
            m_sq = (self.memory_bank ** 2).sum(dim=1).unsqueeze(0)  # (1, N)
            cross = q @ self.memory_bank.T                    # (chunk, N)
            dists = q_sq + m_sq - 2.0 * cross                 # (chunk, N)
            dists = dists.clamp(min=0.0)                       # numerical safety
            min_dist = dists.min(dim=1).values.sqrt()          # (chunk,)
            distances.append(min_dist)

        return torch.cat(distances, dim=0)

    # ------------------------------------------------------------------
    # Predict convenience
    # ------------------------------------------------------------------
    def predict(
        self,
        images: torch.Tensor,
        threshold: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Score images and apply a threshold to produce binary predictions.

        Parameters
        ----------
        images : torch.Tensor
            Batch ``(B, 3, H, W)``.
        threshold : float
            Anomaly score above which an image is labelled defective.

        Returns
        -------
        dict[str, torch.Tensor]
            ``labels``        — ``(B,)`` int, 0 = normal, 1 = defective
            ``anomaly_score`` — ``(B,)`` float
            ``anomaly_map``   — ``(B, 1, H, W)`` float
        """
        result = self.score(images)
        labels = (result["anomaly_score"] > threshold).long()
        result["labels"] = labels
        return result

    @property
    def is_fitted(self) -> bool:
        """Whether the memory bank has been built."""
        return self._is_fitted

    def memory_bank_size(self) -> int:
        """Number of stored patch features."""
        if self.memory_bank is None:
            return 0
        return self.memory_bank.shape[0]
