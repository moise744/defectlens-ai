"""
DefectLens AI — Evaluation script.

Evaluate a trained model on the MVTec AD test set and generate
a comprehensive evaluation report with metrics and visualizations.

Usage:
    python scripts/evaluate.py --config configs/base.yaml --category bottle --checkpoint outputs/bottle/best_model.pth
    python scripts/evaluate.py --config configs/base.yaml --category all --checkpoint-dir outputs/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_category(
    cfg: dict,
    category: str,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """
    Evaluate a trained classifier on one MVTec category.

    Args:
        cfg: Configuration dictionary.
        category: Category name.
        checkpoint_path: Path to model checkpoint.
        output_dir: Directory to save evaluation results.
        device: Compute device.

    Returns:
        Evaluation metrics dictionary.
    """
    from defectlens.data.dataset import MVTecDataset
    from defectlens.data.transforms import get_val_transforms
    from defectlens.evaluation.metrics import compute_classification_metrics
    from defectlens.evaluation.analysis import generate_error_report, save_error_report
    from defectlens.models.classifier import DefectClassifier
    from defectlens.visualization.visualizer import plot_confusion_matrix, plot_roc_curve

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evaluating category: {category.upper()}")
    print(f"{'='*60}")

    # Load dataset
    data_root = Path(cfg["data"]["root"])
    image_size = cfg["data"]["image_size"]
    val_tfms = get_val_transforms(image_size)

    try:
        test_dataset = MVTecDataset(
            root=data_root,
            category=category,
            split="test",
            transform=val_tfms,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load test dataset: {e}")
        return {"error": str(e)}

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg["data"].get("batch_size", 16),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
    )

    # Load model
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("[INFO] Please train a model first using scripts/train.py")
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    model = DefectClassifier(
        backbone=cfg["model"].get("backbone", "resnet18"),
        pretrained=False,
        num_classes=cfg["model"].get("num_classes", 2),
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model = model.to(device)
    model.eval()

    # Run evaluation
    all_labels = []
    all_probs = []
    all_preds = []
    image_paths = []
    defect_types = []

    threshold = cfg.get("evaluation", {}).get("threshold", 0.5)

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            logits, probs = model(images)
            preds = (probs[:, 1] >= threshold).long()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

            if "image_path" in batch:
                image_paths.extend(batch["image_path"])
            if "defect_type" in batch:
                defect_types.extend(batch["defect_type"])

    # Compute metrics
    metrics = compute_classification_metrics(all_labels, all_preds, all_probs)

    print(f"\n[RESULTS] {category}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to: {metrics_path}")

    # Generate visualizations
    if metrics.get("confusion_matrix") is not None:
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names=["Normal", "Defective"],
            save_path=cm_path,
        )
        print(f"[INFO] Confusion matrix saved to: {cm_path}")

    if metrics.get("roc_fpr") is not None:
        roc_path = output_dir / "roc_curve.png"
        plot_roc_curve(
            metrics["roc_fpr"],
            metrics["roc_tpr"],
            auc=metrics["auroc"],
            save_path=roc_path,
        )
        print(f"[INFO] ROC curve saved to: {roc_path}")

    # Error analysis
    if image_paths and defect_types:
        report = generate_error_report(
            labels=all_labels,
            predictions=all_preds,
            probabilities=all_probs,
            image_paths=image_paths,
            categories=defect_types,
        )
        report_path = output_dir / "error_analysis.json"
        save_error_report(report, report_path)
        print(f"[INFO] Error analysis saved to: {report_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DefectLens AI models")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    categories_to_eval = (
        [args.category] if args.category != "all" else [
            "bottle", "cable", "capsule", "carpet", "grid",
            "hazelnut", "leather", "metal_nut", "pill", "screw",
            "tile", "toothbrush", "transistor", "wood", "zipper",
        ]
    )

    all_results = {}
    for cat in categories_to_eval:
        if args.checkpoint:
            ckpt = args.checkpoint
        elif args.checkpoint_dir:
            ckpt = args.checkpoint_dir / cat / "best_model.pth"
        else:
            ckpt = Path(cfg["output"]["dir"]) / cat / "best_model.pth"

        out = args.output or Path(cfg["output"]["dir"]) / cat / "evaluation"
        result = evaluate_category(cfg, cat, ckpt, out, device)
        all_results[cat] = result

    # Save overall summary
    summary_path = Path(cfg["output"]["dir"]) / "evaluation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Evaluation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
