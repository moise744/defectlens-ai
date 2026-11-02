"""Tests for DefectLens inference pipeline."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDefectPredictor:
    """Tests for the inference predictor."""

    def _save_dummy_checkpoint(self, tmp_path: Path) -> Path:
        """Save a minimal model checkpoint for testing."""
        from defectlens.models.classifier import DefectClassifier
        model = DefectClassifier(pretrained=False)
        ckpt_path = tmp_path / "test_model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "val_loss": 0.5,
        }, ckpt_path)
        return ckpt_path

    def test_predictor_loads_checkpoint(self, tmp_path):
        from defectlens.inference.predictor import DefectPredictor
        ckpt = self._save_dummy_checkpoint(tmp_path)
        predictor = DefectPredictor(
            checkpoint_path=ckpt,
            image_size=224,
            threshold=0.5,
            device=torch.device("cpu"),
        )
        assert predictor is not None

    def test_predict_from_numpy_array(self, tmp_path):
        from defectlens.inference.predictor import DefectPredictor
        ckpt = self._save_dummy_checkpoint(tmp_path)
        predictor = DefectPredictor(
            checkpoint_path=ckpt,
            image_size=224,
            threshold=0.5,
            device=torch.device("cpu"),
        )
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predictor.predict(dummy_img)
        assert hasattr(result, "label")
        assert hasattr(result, "confidence")
        assert hasattr(result, "latency_ms")
        assert result.label in (0, 1)
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_ms > 0.0

    def test_predict_from_image_file(self, tmp_path):
        from defectlens.inference.predictor import DefectPredictor
        from PIL import Image
        ckpt = self._save_dummy_checkpoint(tmp_path)
        predictor = DefectPredictor(
            checkpoint_path=ckpt,
            image_size=224,
            threshold=0.5,
            device=torch.device("cpu"),
        )
        # Save a dummy image
        img_path = tmp_path / "test.png"
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(img_path)

        result = predictor.predict(img_path)
        assert result.label in (0, 1)
        assert result.confidence >= 0.0

    def test_invalid_checkpoint_raises(self, tmp_path):
        from defectlens.inference.predictor import DefectPredictor
        fake_path = tmp_path / "nonexistent.pth"
        with pytest.raises((FileNotFoundError, Exception)):
            DefectPredictor(
                checkpoint_path=fake_path,
                image_size=224,
                threshold=0.5,
                device=torch.device("cpu"),
            )

    def test_benchmark_returns_stats(self, tmp_path):
        from defectlens.inference.predictor import DefectPredictor
        ckpt = self._save_dummy_checkpoint(tmp_path)
        predictor = DefectPredictor(
            checkpoint_path=ckpt,
            image_size=224,
            threshold=0.5,
            device=torch.device("cpu"),
        )
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        bench = predictor.benchmark(dummy_img, n_runs=5)
        assert "mean_latency_ms" in bench
        assert "std_latency_ms" in bench
        assert "fps" in bench
        assert "min_latency_ms" in bench
        assert "max_latency_ms" in bench
        assert bench["fps"] > 0


class TestInputValidation:
    """Tests for input validation in the inference pipeline."""

    def test_load_valid_image_numpy(self):
        from defectlens.inference.predictor import load_image
        dummy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = load_image(dummy)
        assert result is not None

    def test_load_image_from_path(self, tmp_path):
        from defectlens.inference.predictor import load_image
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(img_path)
        result = load_image(img_path)
        assert result is not None

    def test_load_nonexistent_image_raises(self):
        from defectlens.inference.predictor import load_image
        with pytest.raises((FileNotFoundError, Exception)):
            load_image(Path("/nonexistent/path/image.png"))
