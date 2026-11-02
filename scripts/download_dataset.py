"""
DefectLens AI — Dataset download and verification script.

Downloads and prepares the MVTec AD dataset for training and evaluation.
Note: MVTec AD requires registration for download. This script provides
verification and organization of a manually downloaded dataset.

Usage:
    python scripts/download_dataset.py --data-dir data/mvtec_ad --verify
    python scripts/download_dataset.py --data-dir data/mvtec_ad --stats
"""

import argparse
import sys
from pathlib import Path


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

DOWNLOAD_URL = "https://www.mvtec.com/company/research/datasets/mvtec-ad"
DOWNLOAD_NOTE = """
================================================================================
MVTec AD DATASET DOWNLOAD INSTRUCTIONS
================================================================================

The MVTec Anomaly Detection dataset requires registration before download.

Steps:
  1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad
  2. Fill in the registration form (name, institution, intended use)
  3. Download the full dataset (~4.9 GB) or individual categories
  4. Extract the archive to: data/mvtec_ad/

Expected directory structure after extraction:
  data/mvtec_ad/
  ├── bottle/
  │   ├── train/
  │   │   └── good/           # Normal training images
  │   ├── test/
  │   │   ├── good/           # Normal test images
  │   │   ├── broken_large/   # Defective images (type name varies)
  │   │   └── ...             # Other defect types
  │   └── ground_truth/
  │       ├── broken_large/   # Pixel-level defect masks
  │       └── ...
  ├── cable/
  │   └── ...
  └── (13 more categories)

License: CC BY-NC-SA 4.0 (Non-Commercial Use Only)
Citation: Bergmann et al., "MVTec AD -- A Comprehensive Real-World Dataset for
          Unsupervised Anomaly Detection", CVPR 2019.

================================================================================
"""


def verify_dataset(data_dir: Path, verbose: bool = True) -> dict:
    """
    Verify dataset structure and integrity.

    Args:
        data_dir: Path to the MVTec AD dataset root directory.
        verbose: Print verification details.

    Returns:
        Dictionary with verification results per category.
    """
    results = {}

    if not data_dir.exists():
        print(f"[ERROR] Dataset directory not found: {data_dir}")
        print("\nRun with --instructions flag to see download instructions.")
        return results

    found_categories = [d.name for d in data_dir.iterdir() if d.is_dir()]
    expected = set(MVTEC_CATEGORIES)
    found = set(found_categories)

    missing = expected - found
    unexpected = found - expected

    if verbose:
        print(f"\n[INFO] Dataset directory: {data_dir.resolve()}")
        print(f"[INFO] Expected categories: {len(expected)}")
        print(f"[INFO] Found categories: {len(found)}")
        if missing:
            print(f"[WARN] Missing categories: {sorted(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected directories: {sorted(unexpected)}")

    for cat in sorted(found & expected):
        cat_dir = data_dir / cat
        train_good = cat_dir / "train" / "good"
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        cat_result = {
            "train_normal": 0,
            "test_normal": 0,
            "test_defective": 0,
            "defect_types": [],
            "has_masks": False,
            "valid": True,
            "errors": [],
        }

        if not train_good.exists():
            cat_result["errors"].append("Missing train/good directory")
            cat_result["valid"] = False
        else:
            cat_result["train_normal"] = len(list(train_good.glob("*.png"))) + \
                                          len(list(train_good.glob("*.jpg")))

        if not test_dir.exists():
            cat_result["errors"].append("Missing test directory")
            cat_result["valid"] = False
        else:
            for split_dir in test_dir.iterdir():
                if not split_dir.is_dir():
                    continue
                count = len(list(split_dir.glob("*.png"))) + len(list(split_dir.glob("*.jpg")))
                if split_dir.name == "good":
                    cat_result["test_normal"] = count
                else:
                    cat_result["test_defective"] += count
                    cat_result["defect_types"].append(split_dir.name)

        cat_result["has_masks"] = gt_dir.exists() and any(gt_dir.iterdir())
        results[cat] = cat_result

        if verbose:
            status = "✓" if cat_result["valid"] else "✗"
            print(
                f"  [{status}] {cat:<15} "
                f"train_normal={cat_result['train_normal']:4d}  "
                f"test_normal={cat_result['test_normal']:4d}  "
                f"test_defective={cat_result['test_defective']:4d}  "
                f"defect_types={len(cat_result['defect_types'])}"
            )

    return results


def print_stats(results: dict) -> None:
    """Print aggregated dataset statistics."""
    if not results:
        print("No verified categories found.")
        return

    total_train = sum(v["train_normal"] for v in results.values())
    total_test_normal = sum(v["test_normal"] for v in results.values())
    total_test_defective = sum(v["test_defective"] for v in results.values())
    total_images = total_train + total_test_normal + total_test_defective

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Categories verified: {len(results)}")
    print(f"Total images:        {total_images}")
    print(f"  Training (normal): {total_train}")
    print(f"  Test (normal):     {total_test_normal}")
    print(f"  Test (defective):  {total_test_defective}")
    print(f"Categories with pixel masks: {sum(1 for v in results.values() if v['has_masks'])}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MVTec AD dataset download instructions and verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mvtec_ad"),
        help="Path to dataset root directory (default: data/mvtec_ad)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset structure and integrity",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics after verification",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print download instructions",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Verify a specific category only",
    )
    args = parser.parse_args()

    if args.instructions or not (args.verify or args.stats):
        print(DOWNLOAD_NOTE)

    if args.verify or args.stats:
        data_dir = args.data_dir
        if args.category:
            if args.category not in MVTEC_CATEGORIES:
                print(f"[ERROR] Unknown category: {args.category}")
                print(f"Valid categories: {MVTEC_CATEGORIES}")
                sys.exit(1)
            data_dir_single = data_dir / args.category
            results = verify_dataset(data_dir, verbose=True)
            results = {k: v for k, v in results.items() if k == args.category}
        else:
            results = verify_dataset(data_dir, verbose=True)

        if args.stats:
            print_stats(results)


if __name__ == "__main__":
    main()
