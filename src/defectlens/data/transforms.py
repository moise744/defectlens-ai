"""
Image transform pipelines for training and evaluation.

Provides configurable torchvision transform compositions for
data augmentation during training and deterministic transforms
for evaluation.
"""

from typing import Optional, Tuple

from torchvision import transforms


def get_train_transforms(
    image_size=256,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Build training transform pipeline with augmentation.

    Args:
        image_size: Target size as int (square) or (height, width) tuple.
        mean: Channel-wise mean for normalization.
        std: Channel-wise standard deviation for normalization.

    Returns:
        A composed transform pipeline.
    """
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transforms(
    image_size: Tuple[int, int] = (256, 256),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Build evaluation transform pipeline (no augmentation).

    Args:
        image_size: Target (height, width) for resizing.
        mean: Channel-wise mean for normalization.
        std: Channel-wise standard deviation for normalization.

    Returns:
        A composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    image_size: int = 256,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Alias for get_eval_transforms with integer image_size support."""
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    return get_eval_transforms(image_size=size, mean=mean, std=std)


def get_mask_transforms(
    image_size: Tuple[int, int] = (256, 256),
) -> transforms.Compose:
    """Build mask transform pipeline (resize + tensor, no normalization).

    Args:
        image_size: Target (height, width) for resizing.

    Returns:
        A composed transform pipeline for ground-truth masks.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
