"""
DefectLens AI — Inference script.

Run inference on a single image or directory of images using a trained model.
Reports prediction, confidence score, and inference latency.

Usage:
    # Single image
    python scripts/inference.py --image path/to/image.png --checkpoint outputs/bottle/best_model.pth

    # Directory
    python scripts/inference.py --image-dir path/to/images/ --checkpoint outputs/bottle/best_model.pth --save-output

    # Benchmark inference speed
    python scripts/inference.py --image path/to/image.png --checkpoint outputs/bottle/best_model.pth --benchmark --n-runs 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def load_config(config_path: Path) -> dict:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DefectLens AI inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", type=Path, default=None, help="Path to a single input image")
    parser.add_argument("--image-dir", type=Path, default=None, help="Path to directory of images")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--image-size", type=int, default=256, help="Input image size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--save-output", action="store_true", help="Save annotated output images")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"), help="Output dir")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--n-runs", type=int, default=50, help="Number of benchmark runs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Provide --image or --image-dir")

    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    # Load model
    from defectlens.models.classifier import DefectClassifier
    from defectlens.inference.predictor import DefectPredictor

    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("[INFO] Train a model first: python scripts/train.py --help")
        sys.exit(1)

    predictor = DefectPredictor(
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        threshold=args.threshold,
        device=device,
    )
    print(f"[INFO] Model loaded from: {args.checkpoint}")

    # Collect images
    if args.image:
        if not args.image.exists():
            print(f"[ERROR] Image not found: {args.image}")
            sys.exit(1)
        images = [args.image]
    else:
        images = [
            p for p in args.image_dir.glob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not images:
            print(f"[ERROR] No supported images found in: {args.image_dir}")
            sys.exit(1)
        print(f"[INFO] Found {len(images)} images")

    # Run benchmark if requested
    if args.benchmark:
        print(f"\n[BENCHMARK] Running {args.n_runs} inference passes on {images[0].name}...")
        bench = predictor.benchmark(images[0], n_runs=args.n_runs)
        print(f"  Mean latency: {bench['mean_latency_ms']:.2f} ms")
        print(f"  Std latency:  {bench['std_latency_ms']:.2f} ms")
        print(f"  FPS:          {bench['fps']:.1f}")
        print(f"  Min latency:  {bench['min_latency_ms']:.2f} ms")
        print(f"  Max latency:  {bench['max_latency_ms']:.2f} ms")

    # Run inference on all images
    if args.save_output:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[INFERENCE RESULTS]")
    print(f"{'Image':<40} {'Prediction':<12} {'Confidence':>10} {'Latency (ms)':>12}")
    print("-" * 78)

    for img_path in sorted(images):
        result = predictor.predict(img_path)
        label_str = "Defective" if result.label == 1 else "Normal"
        print(
            f"{img_path.name:<40} {label_str:<12} {result.confidence:>10.4f} {result.latency_ms:>12.2f}"
        )

        if args.save_output:
            from defectlens.visualization.visualizer import save_image
            import cv2
            import numpy as np

            img = cv2.imread(str(img_path))
            if img is not None:
                color = (0, 0, 255) if result.label == 1 else (0, 200, 0)
                text = f"{label_str}: {result.confidence:.3f}"
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                out_path = args.output_dir / img_path.name
                cv2.imwrite(str(out_path), img)

    if args.save_output:
        print(f"\n[INFO] Annotated images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
