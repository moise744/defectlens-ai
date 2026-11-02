"""
U-Net segmentation model with a pretrained ResNet-18 encoder.

Produces a single-channel binary mask indicating defective regions.
Input sizes must be divisible by 32.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)


class _DecoderBlock(nn.Module):
    """Single decoder block: upsample → concat skip → conv → BN → ReLU × 2."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample *x*, concatenate with *skip*, and apply convolutions."""
        x = self.upsample(x)
        # Handle size mismatches from rounding
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DefectSegmentor(nn.Module):
    """U-Net with a ResNet-18 encoder for binary defect segmentation.

    Parameters
    ----------
    pretrained : bool
        Use ImageNet-pretrained encoder weights.
    freeze_encoder : bool
        If *True*, encoder weights are frozen.

    Notes
    -----
    Input spatial dimensions must be divisible by 32.
    Output is a single-channel sigmoid-activated mask.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        # ── Encoder (ResNet-18) ──────────────────────────────────────
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Stage 0: conv1 + bn1 + relu → 64-ch, H/2
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # MaxPool → H/4
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1   #  64-ch, H/4
        self.enc2 = resnet.layer2   # 128-ch, H/8
        self.enc3 = resnet.layer3   # 256-ch, H/16
        self.enc4 = resnet.layer4   # 512-ch, H/32

        if freeze_encoder:
            for module in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
                for param in module.parameters():
                    param.requires_grad = False
            logger.info("Encoder frozen.")

        # ── Decoder ──────────────────────────────────────────────────
        # dec4: up 512 → concat 256 → 256
        self.dec4 = _DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        # dec3: up 256 → concat 128 → 128
        self.dec3 = _DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        # dec2: up 128 → concat 64 → 64
        self.dec2 = _DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        # dec1: up 64 → concat 64 (enc0) → 64
        self.dec1 = _DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # ── Final upsampling + 1×1 conv ──────────────────────────────
        self.final_upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Segment defect regions.

        Parameters
        ----------
        x : torch.Tensor
            Input batch ``(B, 3, H, W)`` with H, W divisible by 32.

        Returns
        -------
        dict[str, torch.Tensor]
            ``logits`` — raw logits ``(B, 1, H, W)``
            ``mask``   — sigmoid probabilities ``(B, 1, H, W)``
        """
        H, W = x.shape[2], x.shape[3]
        if H % 32 != 0 or W % 32 != 0:
            raise ValueError(
                f"Input size ({H}×{W}) must be divisible by 32. "
                f"Got remainder ({H % 32}, {W % 32})."
            )

        # ── Encoder ──────────────────────────────────────────────────
        s0 = self.enc0(x)         # (B,  64, H/2,  W/2)
        s0p = self.pool(s0)       # (B,  64, H/4,  W/4)
        s1 = self.enc1(s0p)       # (B,  64, H/4,  W/4)
        s2 = self.enc2(s1)        # (B, 128, H/8,  W/8)
        s3 = self.enc3(s2)        # (B, 256, H/16, W/16)
        s4 = self.enc4(s3)        # (B, 512, H/32, W/32)

        # ── Decoder ──────────────────────────────────────────────────
        d4 = self.dec4(s4, s3)    # (B, 256, H/16, W/16)
        d3 = self.dec3(d4, s2)    # (B, 128, H/8,  W/8)
        d2 = self.dec2(d3, s1)    # (B,  64, H/4,  W/4)
        d1 = self.dec1(d2, s0)    # (B,  64, H/2,  W/2)

        # Final upsample to original resolution
        up = self.final_upsample(d1)  # (B, 64, H, W)
        if up.shape[2:] != (H, W):
            up = F.interpolate(up, size=(H, W), mode="bilinear", align_corners=False)
        logits = self.final_conv(up)  # (B, 1, H, W)
        mask = torch.sigmoid(logits)

        return {"logits": logits, "mask": mask}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def predict(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Return a binary mask ``(B, 1, H, W)`` with values in {0, 1}.

        Parameters
        ----------
        x : torch.Tensor
            Input batch.
        threshold : float
            Probability threshold for positive class.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            return (out["mask"] >= threshold).float()

    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def model_summary(self) -> str:
        """Return a summary string."""
        total = self.get_total_params()
        trainable = self.get_trainable_params()
        return (
            f"{'═' * 50}\n"
            f"  DefectSegmentor (U-Net + ResNet-18 Encoder)\n"
            f"{'═' * 50}\n"
            f"  Total params  : {total:,}\n"
            f"  Trainable     : {trainable:,}\n"
            f"  Frozen        : {total - trainable:,}\n"
            f"{'═' * 50}"
        )


def build_segmentor(
    pretrained: bool = True,
    freeze_encoder: bool = False,
    device: str | torch.device = "auto",
) -> DefectSegmentor:
    """Factory: create a segmentor and move it to the target device.

    Parameters
    ----------
    device : str | torch.device
        ``"auto"`` selects CUDA when available.
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = DefectSegmentor(pretrained=pretrained, freeze_encoder=freeze_encoder)
    model = model.to(device)
    logger.info("DefectSegmentor created on %s.", device)
    return model
