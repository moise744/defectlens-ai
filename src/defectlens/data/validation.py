"""
Dataset validation utilities.

Verify the MVTec AD directory structure, check image integrity, report
dataset statistics, and detect corrupted files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)

# Recognised image suffixes (lowercase)
_IMG_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class CategoryStats:
    """Statistics for one product category."""

    name: str
    train_good: int = 0
    test_good: int = 0
    test_defective: int = 0
    defect_types: List[str] = field(default_factory=list)
    mask_count: int = 0
    corrupted_files: List[str] = field(default_factory=list)


@dataclass
class DatasetReport:
    """Full validation report for an MVTec AD dataset root."""

    root: str
    categories: List[CategoryStats] = field(default_factory=list)
    missing_dirs: List[str] = field(default_factory=list)
    corrupted_files: List[str] = field(default_factory=list)
    total_images: int = 0
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines: list[str] = [
            f"{'═' * 60}",
            f"  MVTec AD Dataset Report",
            f"  Root: {self.root}",
            f"{'═' * 60}",
            f"  Valid: {self.is_valid}",
            f"  Categories: {len(self.categories)}",
            f"  Total images: {self.total_images}",
            f"  Corrupted files: {len(self.corrupted_files)}",
        ]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"    • {err}")
        lines.append(f"{'─' * 60}")
        for cat in self.categories:
            lines.append(
                f"  {cat.name:20s}  train_good={cat.train_good:4d}  "
                f"test_good={cat.test_good:4d}  "
                f"test_defective={cat.test_defective:4d}  "
                f"masks={cat.mask_count:4d}  "
                f"defects={cat.defect_types}"
            )
            if cat.corrupted_files:
                for cf in cat.corrupted_files:
                    lines.append(f"    ⚠ corrupted: {cf}")
        lines.append(f"{'═' * 60}")
        return "\n".join(lines)


def _count_images(directory: Path) -> Tuple[int, List[str]]:
    """Count valid image files in *directory*.

    Returns ``(count, list_of_corrupted_paths)``.
    """
    if not directory.is_dir():
        return 0, []
    count = 0
    corrupted: list[str] = []
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() not in _IMG_EXTENSIONS:
            continue
        count += 1
    return count, corrupted


def check_image_integrity(path: Path) -> bool:
    """Return *True* if *path* can be decoded by OpenCV."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return img is not None
    except Exception:
        return False


def check_images_in_dir(directory: Path) -> Tuple[int, List[str]]:
    """Count images and identify corrupted files in *directory*.

    Returns ``(valid_count, corrupted_paths)``.
    """
    if not directory.is_dir():
        return 0, []
    valid = 0
    corrupted: list[str] = []
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() not in _IMG_EXTENSIONS:
            continue
        if check_image_integrity(f):
            valid += 1
        else:
            corrupted.append(str(f))
    return valid, corrupted


def validate_category(
    root: Path,
    category: str,
    *,
    check_integrity: bool = False,
) -> Tuple[CategoryStats, List[str]]:
    """Validate a single category directory.

    Parameters
    ----------
    root : Path
        Dataset root.
    category : str
        Category name.
    check_integrity : bool
        If *True*, each image is opened to verify readability (slow).

    Returns
    -------
    tuple[CategoryStats, list[str]]
        Category statistics and list of structural errors.
    """
    cat_dir = root / category
    stats = CategoryStats(name=category)
    errors: list[str] = []

    # ── train/good ───────────────────────────────────────────────────
    train_good = cat_dir / "train" / "good"
    if not train_good.is_dir():
        errors.append(f"Missing {train_good}")
    else:
        if check_integrity:
            n, bad = check_images_in_dir(train_good)
            stats.train_good = n
            stats.corrupted_files.extend(bad)
        else:
            stats.train_good, _ = _count_images(train_good)

    # ── test/* ───────────────────────────────────────────────────────
    test_dir = cat_dir / "test"
    if not test_dir.is_dir():
        errors.append(f"Missing {test_dir}")
    else:
        for sub in sorted(test_dir.iterdir()):
            if not sub.is_dir():
                continue
            if check_integrity:
                n, bad = check_images_in_dir(sub)
                stats.corrupted_files.extend(bad)
            else:
                n, _ = _count_images(sub)
            if sub.name == "good":
                stats.test_good = n
            else:
                stats.test_defective += n
                stats.defect_types.append(sub.name)

    # ── ground_truth/* ───────────────────────────────────────────────
    gt_dir = cat_dir / "ground_truth"
    if gt_dir.is_dir():
        for sub in sorted(gt_dir.iterdir()):
            if not sub.is_dir():
                continue
            mask_n, _ = _count_images(sub)
            stats.mask_count += mask_n

    return stats, errors


def validate_dataset(
    root: str | Path,
    *,
    check_integrity: bool = False,
    categories: Optional[List[str]] = None,
) -> DatasetReport:
    """Validate the full MVTec AD dataset.

    Parameters
    ----------
    root : str | Path
        Path to the dataset root.
    check_integrity : bool
        If *True*, every image file is opened (slower but thorough).
    categories : list[str], optional
        Restrict validation to these categories.  If *None*, all
        sub-directories that look like categories are checked.

    Returns
    -------
    DatasetReport
        A detailed validation report.
    """
    root = Path(root)
    report = DatasetReport(root=str(root))

    if not root.is_dir():
        report.is_valid = False
        report.errors.append(f"Root directory does not exist: {root}")
        return report

    # Discover categories
    if categories is None:
        categories = sorted(
            d.name
            for d in root.iterdir()
            if d.is_dir() and (d / "train").is_dir()
        )

    if not categories:
        report.is_valid = False
        report.errors.append(
            "No valid categories found. Expected sub-directories with "
            "a 'train' folder."
        )
        return report

    for cat_name in categories:
        cat_stats, cat_errors = validate_category(
            root, cat_name, check_integrity=check_integrity
        )
        report.categories.append(cat_stats)
        report.errors.extend(cat_errors)
        report.corrupted_files.extend(cat_stats.corrupted_files)
        report.total_images += (
            cat_stats.train_good + cat_stats.test_good + cat_stats.test_defective
        )

    if report.errors:
        report.is_valid = False

    return report


def print_report(report: DatasetReport) -> None:
    """Pretty-print a :class:`DatasetReport` to stdout."""
    print(report.summary())
