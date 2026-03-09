"""
Main orchestrator script for the model training pipeline.
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
import yaml
import csv

from src.data.loader import get_data_loaders
from src.model.resnet_152 import ResNet152Model
from src.model.train import ModelTrainer, get_weighted_loss

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str | Path) -> dict:
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_best_model(
    model: torch.nn.Module,
    accuracy: float,
    val_loss: float,
    epoch: int,
    base_dir: str | Path = "artifacts/models",
) -> Path:
    """Saves the model with the best accuracy and validation loss.

    Args:
        model: The trained PyTorch model.
        accuracy: The best validation accuracy achieved.
        val_loss: The validation loss associated with the best accuracy.
        epoch: The epoch at which the best accuracy was achieved.
        base_dir: The base directory where models will be saved.

    Returns:
        The path where the model was saved.
    """
    save_dir = Path(base_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"best_model_ep{epoch}_acc{accuracy:.4f}_loss{val_loss:.4f}.pth"
    save_path = save_dir / filename

    torch.save(model.state_dict(), save_path)
    logging.info(f"Saved best model to {save_path}")

    return save_path


def load_model(model: torch.nn.Module, model_path: str | Path) -> torch.nn.Module:
    """Loads model weights from a specified path.

    Args:
        model: The PyTorch model instance to load weights into.
        model_path: The path to the saved model weights.

    Returns:
        The model with loaded weights.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
        RuntimeError: If there is an issue loading the weights.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")

    try:
        model.load_state_dict(torch.load(path, weights_only=True))
        logging.info(f"Successfully loaded model weights from {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {path}: {e}") from e

    return model


def calculate_class_counts(dataset: torch.utils.data.Dataset, num_classes: int) -> list[int]:
    """Calculates the frequency of each class in the dataset."""
    counts = {i: 0 for i in range(num_classes)}

    import torch.utils.data
    
    # Handle Subset wrapper around ImageFolder
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
        if not hasattr(base_dataset, "samples"):
            raise ValueError("Base dataset does not have 'samples' attribute")
            
        for i in indices:
            _, label = base_dataset.samples[i]
            counts[label] = counts.get(label, 0) + 1
    elif hasattr(dataset, "samples"):
        # Original logic for non-subset ImageFolder
        for _, label in dataset.samples:
            counts[label] = counts.get(label, 0) + 1
    else:
        # Fallback for generic datasets, though slower as it requires basic indexing
        for i in range(len(dataset)):
            _, label = dataset[i]
            counts[label] = counts.get(label, 0) + 1

    # Ensure they are ordered by class index up to num_classes
    ordered_counts = [counts[i] for i in range(num_classes)]
    return ordered_counts


def get_minority_classes(class_counts: list[int], threshold: float = 0.5) -> list[int]:
    """Determines which classes are minority based on the maximum class count.

    Args:
        class_counts: The number of items in each class.
        threshold: The ratio compared to the max class count below which a
                   class is considered a minority.

    Returns:
        A list of class indices considered minority.
    """
    max_count = max(class_counts)
    minority_classes = [
        i for i, count in enumerate(class_counts) if count < max_count * threshold
    ]
    return minority_classes


def main() -> None:
    """Main training execution function."""
    parser = argparse.ArgumentParser(
        description="Train ResNet-152 on Skin Lesion Data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config.yaml file.",
    )
    args = parser.parse_args()

    # Load configurations
    config = load_config(args.config)
    hw_cfg = config["hardware"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    # Device Setup
    use_cuda = hw_cfg.get("use_cuda", True) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Data Loaders
    logging.info("Initializing Data Loaders...")
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir=data_cfg["split_dir"],
        batch_size=train_cfg["batch_size"],
        img_size=tuple(data_cfg["img_size"]),
        num_workers=hw_cfg.get("num_workers", 4),
        subset_fraction=data_cfg.get("subset_fraction", 0.0),
    )
    logging.info(f"Loaded classes: {class_names}")

    # Initialization
    logging.info("Initializing the ResNet-152 Model...")
    model = ResNet152Model(
        num_classes=model_cfg.get("num_classes", len(class_names)),
        pretrained=model_cfg.get("pretrained", True),
    )

    if model_cfg.get("freeze_backbone", True):
        logging.info("Freezing the ResNet-152 backbone...")
        model.freeze_layers()

    # Class Imbalance Handling
    train_dataset = train_loader.dataset
    num_classes = model_cfg.get("num_classes", len(class_names))
    class_counts = calculate_class_counts(train_dataset, num_classes=num_classes)
    logging.info(f"Class distribution: {class_counts}")

    minority_classes = get_minority_classes(class_counts)
    logging.info(f"Minority classes indices: {minority_classes}")

    criterion = get_weighted_loss(class_counts, device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    # Trainer Setup
    trainer = ModelTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        minority_classes=minority_classes,
    )

    # Skip training epoch based on user feedback
    logging.info(
        f"Pipeline setup successfully. Ready to train for {train_cfg['epochs']} epochs."
    )
    logging.info("Skipping actual `trainer.train_loop()` execution per user request.")
    history = trainer.train_loop(train_loader, val_loader, epochs=train_cfg["epochs"])
    
    # Save the model
    best_val_acc = max(history["val_acc"])
    best_epoch = history["val_acc"].index(best_val_acc)
    best_val_loss = history["val_loss"][best_epoch]
    
    save_path = save_best_model(
        model=model,
        accuracy=best_val_acc,
        val_loss=best_val_loss,
        epoch=best_epoch + 1,
    )
    logging.info(f"Model successfully saved to {save_path}")

    # Save the training history to a CSV file
    metrics_dir = Path("artifacts/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path = metrics_dir / "training_history.csv"
    
    with open(history_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Write the header
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        
        # Write the data rows
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.4f}",
                f"{history['train_acc'][i]:.4f}",
                f"{history['val_loss'][i]:.4f}",
                f"{history['val_acc'][i]:.4f}"
            ])
    
    logging.info(f"Training history successfully saved to {history_csv_path}")

if __name__ == "__main__":
    main()
