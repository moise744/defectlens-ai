"""
DefectLens AI — Training script.

Train a binary defect classifier or anomaly detector on the MVTec AD dataset.

Usage:
    # Train classifier on a single category
    python scripts/train.py --config configs/train_classifier.yaml --category bottle

    # Train anomaly detector on all categories
    python scripts/train.py --config configs/train_anomaly.yaml --category all

    # Train with custom output directory
    python scripts/train.py --config configs/train_classifier.yaml --category bottle --output outputs/my_experiment
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import yaml


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg: dict) -> torch.device:
    """Get the compute device based on config and availability."""
    device_cfg = cfg.get("inference", {}).get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_cfg == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")
    return device


def train_classifier(cfg: dict, category: str, output_dir: Path, device: torch.device) -> dict:
    """
    Train binary defect classifier for a single MVTec category.

    Args:
        cfg: Configuration dictionary.
        category: MVTec category name.
        output_dir: Output directory for checkpoints and results.
        device: Compute device.

    Returns:
        Training results dictionary.
    """
    from defectlens.data.dataset import MVTecDataset
    from defectlens.data.transforms import get_train_transforms, get_val_transforms
    from defectlens.data.validation import validate_dataset
    from defectlens.models.classifier import DefectClassifier
    from defectlens.training.trainer import DefectClassifierTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg["data"]["root"])
    image_size = cfg["data"]["image_size"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    print(f"\n{'='*60}")
    print(f"Training category: {category.upper()}")
    print(f"{'='*60}")

    # Validate dataset
    category_dir = data_root / category
    if not category_dir.exists():
        print(f"[ERROR] Category directory not found: {category_dir}")
        print("[INFO] Please download the MVTec AD dataset first:")
        print("       python scripts/download_dataset.py --instructions")
        return {"error": f"Category directory not found: {category_dir}"}

    # Prepare transforms
    train_tfms = get_train_transforms(image_size)
    val_tfms = get_val_transforms(image_size)

    # Create datasets
    try:
        train_dataset = MVTecDataset(
            root=data_root,
            category=category,
            split="train",
            transform=train_tfms,
        )
        val_dataset = MVTecDataset(
            root=data_root,
            category=category,
            split="test",
            transform=val_tfms,
        )
    except Exception as e:
        print(f"[ERROR] Failed to create datasets: {e}")
        return {"error": str(e)}

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Test samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # Initialize model
    model = DefectClassifier(
        backbone=cfg["model"].get("backbone", "resnet18"),
        pretrained=cfg["model"].get("pretrained", True),
        num_classes=cfg["model"].get("num_classes", 2),
    )

    # Initialize trainer
    trainer = DefectClassifierTrainer(
        model=model,
        device=device,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        optimizer_name=cfg["training"]["optimizer"],
        scheduler_name=cfg["training"].get("scheduler", "cosine"),
        early_stopping_patience=cfg["training"].get("early_stopping_patience", 5),
        output_dir=output_dir,
    )

    # Train
    epochs = cfg["training"]["epochs"]
    print(f"[INFO] Starting training for {epochs} epochs...")
    start_time = time.time()

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )

    elapsed = time.time() - start_time
    print(f"\n[INFO] Training completed in {elapsed/60:.1f} minutes")

    # Save training config
    config_save = output_dir / "training_config.yaml"
    with open(config_save, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"[INFO] Results saved to: {output_dir}")
    return {"category": category, "training_time_seconds": elapsed, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DefectLens AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="MVTec category to train on (or 'all' for all categories)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: outputs/{category})",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["classifier", "anomaly"],
        default="classifier",
        help="Model type to train",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Set seed
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)
    print(f"[INFO] Random seed: {seed}")

    # Get device
    device = get_device(cfg)

    # Determine categories to train
    if args.category == "all":
        categories = MVTEC_CATEGORIES
    elif args.category in MVTEC_CATEGORIES:
        categories = [args.category]
    else:
        print(f"[ERROR] Unknown category: {args.category}")
        print(f"[INFO] Valid categories: {MVTEC_CATEGORIES}")
        sys.exit(1)

    # Train each category
    all_results = {}
    for category in categories:
        output_dir = args.output or Path(cfg["output"]["dir"]) / category
        result = train_classifier(cfg, category, output_dir, device)
        all_results[category] = result

    # Save summary
    summary_path = Path(cfg["output"]["dir"]) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[INFO] Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
