"""
Main orchestrator script for the model training pipeline.

Trains ResNet-152 on the HAM10000 skin lesion dataset and logs all
hyperparameters, per-epoch metrics, and the best model checkpoint to
MLflow (backed by DagShub).
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import torch.optim as optim
import yaml
from dotenv import load_dotenv

from src.data.loader import get_data_loaders
from src.model.resnet_152 import ResNet152Model
from src.model.train import ModelTrainer, get_weighted_loss

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    """Loads configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model save / load helpers
# ---------------------------------------------------------------------------

def save_best_model(
    model: torch.nn.Module,
    accuracy: float,
    val_loss: float,
    epoch: int,
    base_dir: str | Path = "artifacts/models",
) -> Path:
    """Saves the model with the best validation accuracy.

    Args:
        model: The trained PyTorch model.
        accuracy: The best validation accuracy achieved.
        val_loss: The validation loss at the best accuracy epoch.
        epoch: The epoch number at which the best accuracy occurred.
        base_dir: Directory where the checkpoint will be written.

    Returns:
        The ``Path`` to the saved checkpoint file.
    """
    save_dir = Path(base_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"best_model_ep{epoch}_acc{accuracy:.4f}_loss{val_loss:.4f}.pth"
    save_path = save_dir / filename

    torch.save(model.state_dict(), save_path)
    logging.info(f"Saved best model to {save_path}")

    return save_path


def load_model(
    model: torch.nn.Module,
    model_path: str | Path,
) -> torch.nn.Module:
    """Loads model weights from a checkpoint file.

    Args:
        model: The PyTorch model instance to load weights into.
        model_path: Path to the saved ``.pth`` checkpoint file.

    Returns:
        The model with the loaded weights applied.

    Raises:
        FileNotFoundError: If no checkpoint exists at *model_path*.
        RuntimeError: If the state-dict cannot be loaded into the model.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")

    try:
        model.load_state_dict(torch.load(path, weights_only=True))
        logging.info(f"Successfully loaded model weights from {path}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model weights from {path}: {exc}"
        ) from exc

    return model


# ---------------------------------------------------------------------------
# Class-imbalance helpers
# ---------------------------------------------------------------------------

def calculate_class_counts(
    dataset: torch.utils.data.Dataset,
    num_classes: int,
) -> list[int]:
    """Calculates the per-class sample frequency in the dataset.

    Handles plain ``ImageFolder`` datasets as well as ``Subset`` wrappers
    produced when ``subset_fraction`` is applied.

    Args:
        dataset: The dataset whose class distribution is computed.
        num_classes: Total number of classes expected.

    Returns:
        A list of length *num_classes* where each element is the count of
        samples belonging to that class index.

    Raises:
        ValueError: If the base dataset of a ``Subset`` has no ``samples``
            attribute.
    """
    import torch.utils.data  # noqa: PLC0415

    counts: dict[int, int] = {i: 0 for i in range(num_classes)}

    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
        if not hasattr(base_dataset, "samples"):
            raise ValueError(
                "Base dataset does not have 'samples' attribute"
            )
        for i in indices:
            _, label = base_dataset.samples[i]
            counts[label] = counts.get(label, 0) + 1
    elif hasattr(dataset, "samples"):
        for _, label in dataset.samples:
            counts[label] = counts.get(label, 0) + 1
    else:
        for i in range(len(dataset)):  # type: ignore[arg-type]
            _, label = dataset[i]
            counts[label] = counts.get(label, 0) + 1

    return [counts[i] for i in range(num_classes)]


def get_minority_classes(
    class_counts: list[int],
    threshold: float = 0.5,
) -> list[int]:
    """Returns indices of classes whose count is below *threshold* × max count.

    Args:
        class_counts: Number of samples per class.
        threshold: Fraction of the max-class count below which a class is
            considered a minority.

    Returns:
        A list of class indices that qualify as minority classes.
    """
    max_count = max(class_counts)
    return [
        i
        for i, count in enumerate(class_counts)
        if count < max_count * threshold
    ]


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _setup_mlflow(config: dict) -> None:
    """Configures the MLflow tracking URI and experiment from env + config.

    The tracking URI is read from the ``MLFLOW_TRACKING_URI`` environment
    variable (set in ``.env``).  The experiment name comes from
    ``config['mlflow']['experiment_name']`` and is suffixed with the current
    date as ``_DD_MM_YYYY``.

    Args:
        config: Full YAML config dict.  Must contain an ``mlflow`` key with
            the sub-key ``experiment_name``.

    Raises:
        KeyError: If the ``mlflow`` section is missing from the config.
        EnvironmentError: If ``MLFLOW_TRACKING_URI`` is not set.
    """
    load_dotenv()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI is not set. Add it to your .env file."
        )
    mlflow_cfg = config["mlflow"]
    date_suffix = datetime.now().strftime("%d_%m_%Y")
    experiment_name = f"{mlflow_cfg['experiment_name']}_{date_suffix}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logging.info(
        f"MLflow tracking URI: {tracking_uri} | "
        f"Experiment: {experiment_name}"
    )


def _log_training_params(
    config: dict,
    class_counts: list[int],
    minority_classes: list[int],
    class_names: list[str],
) -> None:
    """Logs all hyperparameters to the active MLflow run.

    Args:
        config: Full YAML config dict.
        class_counts: Per-class sample counts for the training set.
        minority_classes: Indices of detected minority classes.
        class_names: Ordered list of class name strings.
    """
    hw_cfg = config["hardware"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    mlflow.log_params(
        {
            # hardware
            "use_cuda": hw_cfg.get("use_cuda", True),
            "num_workers": hw_cfg.get("num_workers", 4),
            # model
            "pretrained": model_cfg.get("pretrained", True),
            "num_classes": model_cfg.get("num_classes", 7),
            "freeze_backbone": model_cfg.get("freeze_backbone", True),
            # training
            "batch_size": train_cfg["batch_size"],
            "learning_rate": train_cfg["learning_rate"],
            "epochs": train_cfg["epochs"],
            "weight_decay": train_cfg.get("weight_decay", 0.0),
            "patience": train_cfg.get("patience", 3),
            # data
            "split_dir": data_cfg["split_dir"],
            "img_size": str(data_cfg["img_size"]),
            "subset_fraction": data_cfg.get("subset_fraction", 1.0),
            # class info
            "class_names": str(class_names),
            "class_counts": str(class_counts),
            "minority_classes": str(minority_classes),
        }
    )


def _log_epoch_metrics(history: dict[str, list[float]]) -> None:
    """Logs per-epoch train/val metrics to the active MLflow run.

    Each metric is logged with ``step=epoch_index`` so DagShub renders
    Charts 1 (train acc vs epochs), 2 (val acc vs epochs), and 4
    (train loss vs val loss vs epochs) automatically.

    Args:
        history: Dict with keys ``train_loss``, ``train_acc``,
            ``val_loss``, ``val_acc``, each mapping to a per-epoch list.
    """
    epochs = len(history["train_loss"])
    for epoch in range(epochs):
        mlflow.log_metrics(
            {
                "train_loss": history["train_loss"][epoch],
                "train_acc": history["train_acc"][epoch],
                "val_loss": history["val_loss"][epoch],
                "val_acc": history["val_acc"][epoch],
            },
            step=epoch + 1,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Runs the full training pipeline with MLflow experiment tracking."""
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

    config = load_config(args.config)
    hw_cfg = config["hardware"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    # Device
    use_cuda = hw_cfg.get("use_cuda", True) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Data loaders
    logging.info("Initializing data loaders…")
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir=data_cfg["split_dir"],
        batch_size=train_cfg["batch_size"],
        img_size=tuple(data_cfg["img_size"]),
        num_workers=hw_cfg.get("num_workers", 4),
        subset_fraction=data_cfg.get("subset_fraction", 0.0),
    )
    logging.info(f"Loaded classes: {class_names}")

    # Model
    logging.info("Initializing ResNet-152…")
    model = ResNet152Model(
        num_classes=model_cfg.get("num_classes", len(class_names)),
        pretrained=model_cfg.get("pretrained", True),
    )
    if model_cfg.get("freeze_backbone", True):
        logging.info("Freezing ResNet-152 backbone…")
        model.freeze_layers()

    # Class-imbalance handling
    train_dataset = train_loader.dataset
    num_classes = model_cfg.get("num_classes", len(class_names))
    class_counts = calculate_class_counts(train_dataset, num_classes=num_classes)
    logging.info(f"Class distribution: {class_counts}")
    minority_classes = get_minority_classes(class_counts)
    logging.info(f"Minority class indices: {minority_classes}")
    criterion = get_weighted_loss(class_counts, device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    # Trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        minority_classes=minority_classes,
    )

    # ------------------------------------------------------------------
    # MLflow run
    # ------------------------------------------------------------------
    _setup_mlflow(config)

    with mlflow.start_run():
        logging.info("MLflow run started.")

        # Log all hyperparameters up front
        _log_training_params(config, class_counts, minority_classes, class_names)
        mlflow.set_tag("config_file", args.config)

        # Train
        logging.info(
            f"Starting training for {train_cfg['epochs']} epoch(s)…"
        )
        history = trainer.train_loop(
            train_loader,
            val_loader,
            epochs=train_cfg["epochs"],
            log_mlflow=True,
        )

        # Log per-epoch metrics (Charts 1, 2, 4)
        _log_epoch_metrics(history)

        # Save best checkpoint
        best_val_acc = max(history["val_acc"])
        best_epoch = history["val_acc"].index(best_val_acc) + 1
        best_val_loss = history["val_loss"][best_epoch - 1]

        save_path = save_best_model(
            model=model,
            accuracy=best_val_acc,
            val_loss=best_val_loss,
            epoch=best_epoch,
        )
        logging.info(f"Best checkpoint saved to {save_path}")

        # Log the checkpoint as an MLflow artifact
        mlflow.log_artifact(str(save_path), artifact_path="model")
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.set_tag("best_checkpoint", str(save_path))

        logging.info("MLflow run ended successfully.")


if __name__ == "__main__":
    main()
