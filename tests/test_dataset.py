"""Tests for DefectLens dataset module."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestTransforms:
    """Test data augmentation transforms."""

    def test_val_transforms_output_shape(self):
        from defectlens.data.transforms import get_val_transforms
        from PIL import Image
        tfms = get_val_transforms(image_size=256)
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        tensor = tfms(img)
        assert tensor.shape == (3, 256, 256), f"Expected (3,256,256), got {tensor.shape}"

    def test_train_transforms_output_shape(self):
        from defectlens.data.transforms import get_train_transforms
        from PIL import Image
        tfms = get_train_transforms(image_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        tensor = tfms(img)
        assert tensor.shape == (3, 224, 224), f"Expected (3,224,224), got {tensor.shape}"

    def test_train_transforms_different_from_val(self):
        """Train and val transforms should be different objects."""
        from defectlens.data.transforms import get_train_transforms, get_val_transforms
        train_t = get_train_transforms(256)
        val_t = get_val_transforms(256)
        assert train_t is not val_t

    def test_normalization_range(self):
        """Normalized output should not be in 0-255 range."""
        from defectlens.data.transforms import get_val_transforms
        from PIL import Image
        tfms = get_val_transforms(256)
        img = Image.fromarray(np.full((256, 256, 3), 255, dtype=np.uint8))
        tensor = tfms(img)
        # After ImageNet normalization, max white pixel should be around 2.6 (not 255)
        assert tensor.max().item() < 5.0

    def test_output_is_tensor(self):
        from defectlens.data.transforms import get_val_transforms
        from PIL import Image
        tfms = get_val_transforms(256)
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = tfms(img)
        assert isinstance(result, torch.Tensor)


class TestDataValidation:
    """Test dataset validation utilities."""

    def test_validate_nonexistent_directory(self, tmp_path):
        from defectlens.data.validation import validate_dataset
        non_existent = tmp_path / "does_not_exist"
        report = validate_dataset(non_existent)
        assert report.is_valid is False
        assert len(report.errors) > 0

    def test_validate_empty_directory(self, tmp_path):
        from defectlens.data.validation import validate_dataset
        (tmp_path / "bottle").mkdir()
        report = validate_dataset(tmp_path)
        # Should report missing subdirectories/structure
        assert report.is_valid is False

    def test_validate_correct_structure(self, tmp_path):
        from defectlens.data.validation import validate_dataset
        from PIL import Image

        # Create minimal valid MVTec-like structure
        cat_dir = tmp_path / "bottle"
        train_good = cat_dir / "train" / "good"
        test_good = cat_dir / "test" / "good"
        test_broken = cat_dir / "test" / "broken_large"
        gt_broken = cat_dir / "ground_truth" / "broken_large"

        for d in [train_good, test_good, test_broken, gt_broken]:
            d.mkdir(parents=True)

        # Create dummy images
        dummy = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        for i in range(3):
            dummy.save(train_good / f"img_{i:03d}.png")
            dummy.save(test_good / f"img_{i:03d}.png")
        dummy.save(test_broken / "img_001.png")
        dummy.save(gt_broken / "img_001.png")

        report = validate_dataset(tmp_path, categories=["bottle"])
        assert report.is_valid is True
        cat_stats = report.categories[0]
        assert cat_stats.train_good == 3
        assert cat_stats.test_good == 3
        assert cat_stats.test_defective == 1


class TestMVTecDataset:
    """Test the MVTecDataset class with mock data."""

    def _create_mock_dataset(self, root: Path, category: str = "bottle"):
        """Create minimal MVTec-like directory for testing."""
        from PIL import Image
        cat_dir = root / category
        train_good = cat_dir / "train" / "good"
        test_good = cat_dir / "test" / "good"
        test_broken = cat_dir / "test" / "broken_large"
        gt_broken = cat_dir / "ground_truth" / "broken_large"
        for d in [train_good, test_good, test_broken, gt_broken]:
            d.mkdir(parents=True, exist_ok=True)
        dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        dummy_mask = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        for i in range(5):
            dummy.save(train_good / f"img_{i:03d}.png")
        for i in range(3):
            dummy.save(test_good / f"img_{i:03d}.png")
        dummy.save(test_broken / "img_001.png")
        dummy_mask.save(gt_broken / "img_001.png")
        return root

    def test_train_dataset_length(self, tmp_path):
        from defectlens.data.dataset import MVTecDataset
        from defectlens.data.transforms import get_val_transforms
        self._create_mock_dataset(tmp_path)
        ds = MVTecDataset(root=tmp_path, category="bottle", split="train",
                          transform=get_val_transforms(64))
        assert len(ds) == 5

    def test_test_dataset_length(self, tmp_path):
        from defectlens.data.dataset import MVTecDataset
        from defectlens.data.transforms import get_val_transforms
        self._create_mock_dataset(tmp_path)
        ds = MVTecDataset(root=tmp_path, category="bottle", split="test",
                          transform=get_val_transforms(64))
        assert len(ds) == 4  # 3 good + 1 broken

    def test_dataset_item_structure(self, tmp_path):
        from defectlens.data.dataset import MVTecDataset
        from defectlens.data.transforms import get_val_transforms
        self._create_mock_dataset(tmp_path)
        ds = MVTecDataset(root=tmp_path, category="bottle", split="train",
                          transform=get_val_transforms(64))
        item = ds[0]
        assert "image" in item
        assert "label" in item
        assert isinstance(item["image"], torch.Tensor)
        assert item["image"].shape == (3, 64, 64)
        assert item["label"] in (0, 1)

    def test_train_labels_all_normal(self, tmp_path):
        from defectlens.data.dataset import MVTecDataset
        from defectlens.data.transforms import get_val_transforms
        self._create_mock_dataset(tmp_path)
        ds = MVTecDataset(root=tmp_path, category="bottle", split="train",
                          transform=get_val_transforms(64))
        for i in range(len(ds)):
            assert ds[i]["label"] == 0, "Training images should all be normal (label=0)"

    def test_defective_item_has_label_one(self, tmp_path):
        from defectlens.data.dataset import MVTecDataset
        from defectlens.data.transforms import get_val_transforms
        self._create_mock_dataset(tmp_path)
        ds = MVTecDataset(root=tmp_path, category="bottle", split="test",
                          transform=get_val_transforms(64))
        labels = [ds[i]["label"] for i in range(len(ds))]
        assert 1 in labels, "Test set should contain at least one defective image (label=1)"
