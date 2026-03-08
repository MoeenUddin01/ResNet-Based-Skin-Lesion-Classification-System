"""
Main orchestrator script for the model training pipeline.
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
import yaml

from src.data.loader import get_data_loaders
from src.model.resnet_152 import ResNet152Model
from src.model.train import ModelTrainer, get_weighted_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: str | Path) -> dict:
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_class_counts(dataset: torch.utils.data.Dataset) -> list[int]:
    """Calculates the frequency of each class in the dataset."""
    counts = {}
    for _, label in dataset.samples:
        counts[label] = counts.get(label, 0) + 1
    
    # Ensure they are ordered by class index
    ordered_counts = [counts[i] for i in range(len(counts))]
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
    parser = argparse.ArgumentParser(description="Train ResNet-152 on Skin Lesion Data.")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config.yaml file."
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
    class_counts = calculate_class_counts(train_dataset)
    logging.info(f"Class distribution: {class_counts}")

    minority_classes = get_minority_classes(class_counts)
    logging.info(f"Minority classes indices: {minority_classes}")

    criterion = get_weighted_loss(class_counts, device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=train_cfg["learning_rate"], 
        weight_decay=train_cfg.get("weight_decay", 0.0)
    )

    # Trainer Setup
    trainer = ModelTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        minority_classes=minority_classes
    )

    # Skip training epoch based on user feedback
    logging.info(
        f"Pipeline setup successfully. Ready to train for {train_cfg['epochs']} epochs."
    )
    logging.info("Skipping actual `trainer.train_loop()` execution per user request.")
    # trainer.train_loop(train_loader, val_loader, epochs=train_cfg["epochs"])


if __name__ == "__main__":
    main()
