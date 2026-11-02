"""
DefectLens AI - Training Engine.

Provides a configurable training loop for defect classification models
with support for early stopping, metric tracking, and checkpoint saving.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: One of 'min' (lower is better) or 'max' (higher is better).
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.should_stop: bool = False

    def __call__(self, metric: float) -> bool:
        """Check whether training should stop.

        Args:
            metric: Current epoch metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        improved = (
            (metric < self.best_score - self.min_delta)
            if self.mode == "min"
            else (metric > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


class TrainingHistory:
    """Container for tracking training metrics across epochs."""

    def __init__(self) -> None:
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_acc: List[float] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []

    def record(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
        epoch_time: float,
    ) -> None:
        """Record metrics for one epoch.

        Args:
            train_loss: Training loss.
            val_loss: Validation loss.
            train_acc: Training accuracy.
            val_acc: Validation accuracy.
            lr: Current learning rate.
            epoch_time: Time taken for the epoch in seconds.
        """
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

    def to_dict(self) -> Dict[str, Any]:
        """Export history as a serializable dictionary."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "total_epochs": len(self.train_loss),
            "best_val_loss": min(self.val_loss) if self.val_loss else None,
            "best_val_acc": max(self.val_acc) if self.val_acc else None,
        }

    def save(self, path: Path) -> None:
        """Save training history to a JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Training history saved to %s", path)


class DefectClassifierTrainer:
    """Training engine for the DefectLens binary classifier.

    Supports configurable optimizers, schedulers, early stopping,
    and automatic checkpoint management.

    Args:
        model: PyTorch model to train.
        config: Training configuration dictionary with keys such as
            ``epochs``, ``learning_rate``, ``weight_decay``, ``optimizer``,
            ``scheduler``, ``early_stopping_patience``.
        output_dir: Directory for saving checkpoints and history.
        device: Device string (``"auto"``, ``"cpu"``, ``"cuda"``).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        output_dir: str = "outputs",
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        # Reproducibility
        set_seed(seed)

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info("Training on device: %s", self.device)

        self.model = model.to(self.device)
        self.config = config

        # Output paths
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training hyperparameters
        self.epochs: int = config.get("epochs", 20)
        self.lr: float = config.get("learning_rate", 1e-3)
        self.weight_decay: float = config.get("weight_decay", 1e-4)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._build_optimizer(config.get("optimizer", "adam"))

        # Scheduler
        self.scheduler = self._build_scheduler(config.get("scheduler", "cosine"))

        # Early stopping
        patience = config.get("early_stopping_patience", 5)
        self.early_stopping = EarlyStopping(patience=patience, mode="min") if patience > 0 else None

        # History tracker
        self.history = TrainingHistory()

        # Best model tracking
        self.best_val_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self, name: str) -> torch.optim.Optimizer:
        """Create optimizer from name string.

        Args:
            name: One of ``"adam"``, ``"adamw"``, ``"sgd"``.

        Returns:
            Configured optimizer instance.

        Raises:
            ValueError: If the optimizer name is unrecognised.
        """
        name_lower = name.lower()
        params = self.model.parameters()
        if name_lower == "adam":
            return Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if name_lower == "adamw":
            return AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if name_lower == "sgd":
            return SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        raise ValueError(f"Unsupported optimizer: '{name}'. Choose from adam, adamw, sgd.")

    def _build_scheduler(self, name: str) -> Optional[Any]:
        """Create learning-rate scheduler from name string.

        Args:
            name: One of ``"cosine"``, ``"step"``, ``"plateau"``, ``"none"``.

        Returns:
            Scheduler instance or ``None``.

        Raises:
            ValueError: If the scheduler name is unrecognised.
        """
        name_lower = name.lower()
        if name_lower == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        if name_lower == "step":
            return StepLR(self.optimizer, step_size=max(1, self.epochs // 3), gamma=0.1)
        if name_lower == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="min", patience=3, factor=0.5)
        if name_lower == "none":
            return None
        raise ValueError(f"Unsupported scheduler: '{name}'. Choose from cosine, step, plateau, none.")

    # ------------------------------------------------------------------
    # Training / validation loops
    # ------------------------------------------------------------------

    def _train_one_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Run one validation pass.

        Args:
            dataloader: Validation data loader.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_acc: float,
        is_best: bool = False,
    ) -> Path:
        """Save a model checkpoint.

        Args:
            epoch: Current epoch number (1-indexed).
            val_loss: Validation loss.
            val_acc: Validation accuracy.
            is_best: Whether this is the best model so far.

        Returns:
            Path to the saved checkpoint.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": self.config,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch:03d}.pth"
        path = self.checkpoint_dir / filename
        torch.save(state, path)
        logger.info("Checkpoint saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        """Execute the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.

        Returns:
            A :class:`TrainingHistory` instance with recorded metrics.
        """
        logger.info(
            "Starting training: %d epochs | lr=%.1e | device=%s",
            self.epochs, self.lr, self.device,
        )

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self._train_one_epoch(train_loader)

            # Validate
            val_loss, val_acc = self._validate(val_loader)

            # Current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Record history
            self.history.record(train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)

            # Logging
            logger.info(
                "Epoch %3d/%d | Train Loss: %.4f Acc: %.4f | "
                "Val Loss: %.4f Acc: %.4f | LR: %.2e | Time: %.1fs",
                epoch, self.epochs,
                train_loss, train_acc,
                val_loss, val_acc,
                current_lr, epoch_time,
            )

            # Checkpoint best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=True)

            # Early stopping
            if self.early_stopping is not None and self.early_stopping(val_loss):
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d)",
                    epoch, self.early_stopping.patience,
                )
                break

        # Save final checkpoint and history
        self._save_checkpoint(epoch, val_loss, val_acc, is_best=False)
        self.history.save(self.output_dir / "training_history.json")

        logger.info(
            "Training complete. Best val loss: %.4f | Best val acc: %.4f",
            self.best_val_loss,
            max(self.history.val_acc) if self.history.val_acc else 0.0,
        )
        return self.history
