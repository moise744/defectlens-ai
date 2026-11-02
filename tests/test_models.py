"""Tests for DefectLens model architectures."""
import sys
from pathlib import Path
import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def dummy_image_batch():
    """Create a batch of dummy images for testing."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def dummy_single_image():
    return torch.randn(1, 3, 256, 256)


class TestDefectClassifier:
    """Tests for the binary defect classifier."""

    def test_instantiation_default(self):
        from defectlens.models.classifier import DefectClassifier
        model = DefectClassifier(pretrained=False)
        assert model is not None

    def test_instantiation_resnet18(self):
        from defectlens.models.classifier import DefectClassifier
        model = DefectClassifier(mode="finetune", pretrained=False, num_classes=2)
        assert model is not None

    def test_forward_output_shape_logits(self, dummy_image_batch):
        from defectlens.models.classifier import DefectClassifier
        model = DefectClassifier(mode="finetune", pretrained=False, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image_batch)
        logits = out["logits"]
        assert logits.shape == (2, 2), f"Expected (2,2), got {logits.shape}"

    def test_forward_output_shape_probs(self, dummy_image_batch):
        from defectlens.models.classifier import DefectClassifier
        model = DefectClassifier(mode="finetune", pretrained=False, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image_batch)
        probs = out["probs"]
        assert probs.shape == (2, 2), f"Expected (2,2), got {probs.shape}"

    def test_probabilities_sum_to_one(self, dummy_image_batch):
        from defectlens.models.classifier import DefectClassifier
        model = DefectClassifier(mode="finetune", pretrained=False, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image_batch)
        probs = out["probs"]
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5), "Probabilities should sum to 1"

    def test_probabilities_in_range(self, dummy_image_batch):
        from defectlens.models.classifier import DefectClassifier
        model = DefectClassifier(mode="finetune", pretrained=False, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image_batch)
        probs = out["probs"]
        assert (probs >= 0).all() and (probs <= 1).all(), "Probs should be in [0, 1]"

    def test_cpu_execution(self):
        from defectlens.models.classifier import DefectClassifier
        model = DefectClassifier(mode="finetune", pretrained=False)
        model = model.to(torch.device("cpu"))
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out["logits"].device.type == "cpu"

    def test_different_input_sizes(self):
        from defectlens.models.classifier import DefectClassifier
        model = DefectClassifier(mode="finetune", pretrained=False)
        model.eval()
        for size in [128, 224, 256, 320]:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                out = model(x)
            assert out["logits"].shape == (1, 2)


class TestPatchCoreAnomalyDetector:
    """Tests for the PatchCore anomaly detection model."""

    def test_instantiation(self):
        from defectlens.models.anomaly import PatchCoreAnomalyDetector
        model = PatchCoreAnomalyDetector(device="cpu", pretrained=False)
        assert model is not None
        assert not model.is_fitted

    def test_fit_and_score(self):
        from defectlens.models.anomaly import PatchCoreAnomalyDetector
        from torch.utils.data import DataLoader, TensorDataset

        model = PatchCoreAnomalyDetector(device="cpu", max_memory_samples=100, pretrained=False)

        # Mock normal training dataset
        dummy_imgs = torch.randn(4, 3, 64, 64)
        dummy_dataset = [{"image": dummy_imgs[i]} for i in range(4)]
        train_loader = DataLoader(dummy_dataset, batch_size=2)

        model.fit(train_loader)
        assert model.is_fitted
        assert model.memory_bank_size() > 0

        # Score test image
        test_batch = torch.randn(2, 3, 64, 64)
        results = model.score(test_batch)

        assert "anomaly_score" in results
        assert "anomaly_map" in results
        assert results["anomaly_score"].shape == (2,)
        assert results["anomaly_map"].shape == (2, 1, 64, 64)

    def test_predict_with_threshold(self):
        from defectlens.models.anomaly import PatchCoreAnomalyDetector
        from torch.utils.data import DataLoader

        model = PatchCoreAnomalyDetector(device="cpu", max_memory_samples=50, pretrained=False)
        dummy_dataset = [{"image": torch.randn(3, 64, 64)} for _ in range(2)]
        train_loader = DataLoader(dummy_dataset, batch_size=2)
        model.fit(train_loader)

        test_batch = torch.randn(2, 3, 64, 64)
        preds = model.predict(test_batch, threshold=0.5)

        assert "labels" in preds
        assert preds["labels"].shape == (2,)
        assert set(preds["labels"].tolist()).issubset({0, 1})


class TestDefectSegmentor:
    """Tests for the U-Net defect segmentation model."""

    def test_instantiation(self):
        from defectlens.models.segmentor import DefectSegmentor
        model = DefectSegmentor(pretrained=False)
        assert model is not None

    def test_output_shape_matches_input(self):
        from defectlens.models.segmentor import DefectSegmentor
        model = DefectSegmentor(pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        mask = out["mask"]
        assert mask.shape == (1, 1, 256, 256), f"Expected (1,1,256,256), got {mask.shape}"

    def test_output_in_0_1_range(self):
        from defectlens.models.segmentor import DefectSegmentor
        model = DefectSegmentor(pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        mask = out["mask"]
        assert (mask >= 0).all() and (mask <= 1).all(), "Sigmoid mask should be in [0, 1]"

    def test_batch_processing(self):
        from defectlens.models.segmentor import DefectSegmentor
        model = DefectSegmentor(pretrained=False)
        model.eval()
        x = torch.randn(4, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out["mask"].shape[0] == 4, "Batch dimension should be preserved"
