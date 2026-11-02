"""
Dataset classes for MVTec AD and generic defect detection.

Supports loading images from the MVTec Anomaly Detection dataset
with proper train/test splits and ground-truth masks.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from defectlens.data.transforms import get_eval_transforms, get_mask_transforms


class MVTecDataset(Dataset):
    """MVTec Anomaly Detection dataset.

    Expects the standard MVTec AD directory layout::

        <root>/<category>/train/good/
        <root>/<category>/test/good/
        <root>/<category>/test/<defect_type>/
        <root>/<category>/ground_truth/<defect_type>/

    Args:
        root: Root directory of the MVTec AD dataset.
        category: Product category (e.g. ``"bottle"``, ``"cable"``).
        split: ``"train"`` or ``"test"``.
        transform: Optional transform applied to images.
        mask_transform: Optional transform applied to masks.
    """

    CATEGORIES: List[str] = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ]

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ) -> None:
        if category not in self.CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Available: {self.CATEGORIES}"
            )
        if split not in ("train", "test"):
            raise ValueError(f"Split must be 'train' or 'test', got '{split}'.")

        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform or get_eval_transforms()
        self.mask_transform = mask_transform or get_mask_transforms()

        self.category_dir = self.root / category
        self.split_dir = self.category_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.split_dir}. "
                f"Please download MVTec AD first."
            )

        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Scan the split directory and build the sample list."""
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        for defect_dir in sorted(self.split_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name
            is_normal = defect_type == "good"
            label = 0 if is_normal else 1

            for img_path in sorted(defect_dir.iterdir()):
                if img_path.suffix.lower() not in image_extensions:
                    continue

                mask_path = None
                if not is_normal and self.split == "test":
                    mask_candidate = (
                        self.category_dir / "ground_truth" / defect_type
                        / img_path.name.replace(img_path.suffix, "_mask" + img_path.suffix)
                    )
                    # Try alternate naming conventions
                    if not mask_candidate.exists():
                        mask_candidate = (
                            self.category_dir / "ground_truth" / defect_type
                            / (img_path.stem + "_mask.png")
                        )
                    if mask_candidate.exists():
                        mask_path = mask_candidate

                self.samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "label": label,
                    "defect_type": defect_type,
                    "category": self.category,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index.

        Returns:
            Dictionary with keys: ``image``, ``label``, ``mask``,
            ``defect_type``, ``image_path``.
        """
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        mask = torch.zeros(1, image.shape[1], image.shape[2])
        if sample["mask_path"] is not None:
            mask_img = Image.open(sample["mask_path"]).convert("L")
            mask = self.mask_transform(mask_img)
            mask = (mask > 0.5).float()

        return {
            "image": image,
            "label": sample["label"],
            "mask": mask,
            "defect_type": sample["defect_type"],
            "image_path": str(sample["image_path"]),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        labels = [s["label"] for s in self.samples]
        defect_types = [s["defect_type"] for s in self.samples]
        unique_defects = sorted(set(defect_types))
        return {
            "category": self.category,
            "split": self.split,
            "total_samples": len(self.samples),
            "normal_samples": labels.count(0),
            "defective_samples": labels.count(1),
            "defect_types": unique_defects,
        }


class DefectLensDataset(Dataset):
    """Generic defect detection dataset.

    Loads images from a flat or nested directory. Images in a folder
    named ``good`` or ``normal`` are labelled 0; all others are labelled 1.

    Args:
        root: Directory containing images (possibly in sub-folders).
        transform: Optional transform applied to images.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.transform = transform or get_eval_transforms()
        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Discover image files and assign labels."""
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        normal_names = {"good", "normal", "ok"}

        for img_path in sorted(self.root.rglob("*")):
            if img_path.suffix.lower() not in image_extensions:
                continue
            parent_name = img_path.parent.name.lower()
            label = 0 if parent_name in normal_names else 1
            self.samples.append({
                "image_path": img_path,
                "label": label,
                "defect_type": parent_name,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "label": sample["label"],
            "mask": torch.zeros(1, image.shape[1], image.shape[2]),
            "defect_type": sample["defect_type"],
            "image_path": str(sample["image_path"]),
        }
