"""
Model training utilities for skin lesion classification.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_weighted_loss(class_counts: list[int], device: torch.device) -> nn.CrossEntropyLoss:
    """Computes a weighted cross-entropy loss to handle class imbalance.

    Weights are calculated as the inverse of the class frequencies.

    Args:
        class_counts: A list containing the number of samples for each class.
        device: The device to place the weight tensor on.

    Returns:
        A CrossEntropyLoss configured with class weights.
    """
    total_samples = sum(class_counts)
    # Inverse frequency weighting. Handle division by zero if count is 0.
    weights = [
        total_samples / (len(class_counts) * count) if count > 0 else 0.0
        for count in class_counts
    ]
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    
    logging.info(f"Initialized weighted loss with weights: {weights}")
    return nn.CrossEntropyLoss(weight=weight_tensor)


def apply_minority_augmentation(
    images: torch.Tensor,
    labels: torch.Tensor,
    minority_classes: list[int],
    augmentation_transform: Any
) -> torch.Tensor:
    """Applies on-the-fly augmentation to images of minority classes in a batch.

    Args:
        images: Batch of image tensors.
        labels: Batch of labels.
        minority_classes: A list of class indices considered as minority.
        augmentation_transform: A torchvision transform to apply.

    Returns:
        The augmented batch of images.
    """
    augmented_images = images.clone()
    for i in range(len(labels)):
        if labels[i].item() in minority_classes:
            # Apply augmentation (requires images to not be strictly batched or handled properly)
            # Since torchvision transforms usually expect PIL or untrimmed tensors, 
            # we rely on transforms that support C,H,W tensors directly.
            augmented_images[i] = augmentation_transform(augmented_images[i])
            
    return augmented_images


class ModelTrainer:
    """Handles the training loop for the skin lesion classification model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        minority_classes: list[int] | None = None,
    ) -> None:
        """Initializes the ModelTrainer.

        Args:
            model: The neural network model to train.
            device: The device to run training upon (CPU or CUDA).
            criterion: The loss function (e.g., CrossEntropyLoss).
            optimizer: The optimizer used to update model weights.
            minority_classes: List of minority class indices for targeted augmentation.
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.minority_classes = minority_classes or []

        # Simple batched augmentation supporting tensors for minority classes
        self.minority_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # Add further tensor-compatible augmentations if needed
        ])

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """Runs a single epoch of training.

        Args:
            dataloader: DataLoader providing the training batches.

        Returns:
            A tuple of (average_loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        pbar = tqdm(dataloader, desc="Training Batch", leave=False)
        for images, labels in pbar:
            if self.minority_classes:
                images = apply_minority_augmentation(
                    images, labels, self.minority_classes, self.minority_transform
                )

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data).item()
            total_samples += images.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = correct_preds / total_samples
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        """Evaluates the model on a validation or test set.

        Args:
            dataloader: DataLoader providing the evaluation batches.

        Returns:
            A tuple of (average_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation Batch", leave=False)
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels.data).item()
                total_samples += images.size(0)

        eval_loss = total_loss / total_samples
        eval_acc = correct_preds / total_samples
        return eval_loss, eval_acc

    def train_loop(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> dict[str, list[float]]:
        """Executes the full training and validation loop over multiple epochs.

        Args:
            train_loader: The DataLoader for the training set.
            val_loader: The DataLoader for the validation set.
            epochs: The number of epochs to train.

        Returns:
            A dictionary containing historical loss and accuracy lists.
        """
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }

        logging.info(f"Starting training loop for {epochs} epochs on device: {self.device}")

        best_val_acc = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            import mlflow
            if mlflow.active_run():
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    step=epoch + 1,
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_model.pth")
                logging.info(f"New best model saved! Epoch: {epoch+1}, Val Acc: {val_acc:.4f}")

            logging.info(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        logging.info("Training loop completed.")
        
        import mlflow
        if mlflow.active_run() and os.path.exists("best_model.pth"):
            mlflow.log_artifact("best_model.pth", artifact_path="model")

        return history
