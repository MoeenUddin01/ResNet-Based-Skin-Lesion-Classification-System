"""
Model evaluation pipeline script for skin lesion classification.

Loads a trained checkpoint, evaluates on the held-out test set, and persists
per-class precision/recall/F1 together with overall accuracy and loss to
artifacts/testing_result/.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.loader import get_test_loader
from src.model.resnet_152 import ResNet152Model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str | Path) -> dict:
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration values.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ModelEvaluator:
    """Handles loading, evaluating and persisting results for a trained model.

    Attributes:
        config: Full configuration dictionary loaded from YAML.
        device: The torch device (CPU or CUDA) that inference will run on.
        model: The ResNet-152 classification model.
        test_loader: DataLoader for the held-out test set.
        class_names: Ordered list of class name strings.
        results_dir: Directory where result CSVs are written.
    """

    def __init__(self, config: dict) -> None:
        """Initialises the evaluator from a configuration dictionary.

        Args:
            config: A dictionary with keys ``hardware``, ``model``, and
                ``data``.  The ``data`` section must contain ``test_dir``
                and ``img_size``.
        """
        self.config = config
        hw_cfg = config["hardware"]
        use_cuda = hw_cfg.get("use_cuda", True) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logging.info(f"Evaluator using device: {self.device}")

        self.model: nn.Module | None = None
        self.test_loader: DataLoader | None = None
        self.class_names: list[str] = []
        self.results_dir = Path("artifacts/testing_result")

    # ------------------------------------------------------------------
    # 1. Load the model
    # ------------------------------------------------------------------
    def load_model(self, model_path: str | Path) -> None:
        """Loads a trained ResNet-152 checkpoint from disk.

        Args:
            model_path: Path to the ``.pth`` checkpoint file saved by the
                training pipeline.

        Raises:
            FileNotFoundError: If no checkpoint exists at *model_path*.
            RuntimeError: If the state-dict cannot be loaded into the model.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        model_cfg = self.config["model"]
        num_classes = model_cfg.get("num_classes", 7)

        model = ResNet152Model(num_classes=num_classes, pretrained=False)

        try:
            model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load weights from {path}: {exc}") from exc

        model.to(self.device)
        model.eval()
        self.model = model
        logging.info(f"Loaded model weights from {path}")

    # ------------------------------------------------------------------
    # 2. Load the test dataset
    # ------------------------------------------------------------------
    def load_test_dataset(self) -> None:
        """Loads the test DataLoader using the path configured in ``data``.

        The ``data`` section of the config must contain ``test_dir``
        (root directory whose ``test/`` subdirectory holds class folders)
        and ``img_size``.

        Raises:
            KeyError: If the ``test_dir`` key is missing from the config.
            ValueError: If the test directory does not exist on disk.
        """
        data_cfg = self.config["data"]
        if "test_dir" not in data_cfg:
            raise KeyError(
                "'test_dir' is missing from the 'data' section of the config."
            )

        hw_cfg = self.config["hardware"]
        test_loader, class_names = get_test_loader(
            data_dir=data_cfg["test_dir"],
            batch_size=self.config["training"].get("batch_size", 32),
            img_size=tuple(data_cfg["img_size"]),
            num_workers=hw_cfg.get("num_workers", 4),
        )
        self.test_loader = test_loader
        self.class_names = class_names
        logging.info(
            f"Test dataset loaded: {len(test_loader.dataset)} samples, "  # type: ignore[arg-type]
            f"classes: {class_names}"
        )

    # ------------------------------------------------------------------
    # 3. Evaluation logic
    # ------------------------------------------------------------------
    def evaluate(self) -> dict[str, object]:
        """Runs inference over the test set and computes metrics.

        Computes overall accuracy and loss as well as per-class precision,
        recall, and F1-score.

        Returns:
            A dictionary with keys:
                - ``overall_accuracy`` (float)
                - ``overall_loss`` (float)
                - ``per_class`` (list[dict]) — one dict per class with keys
                  ``class_name``, ``precision``, ``recall``, ``f1``,
                  ``support``.

        Raises:
            RuntimeError: If the model or test loader has not been loaded yet.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.test_loader is None:
            raise RuntimeError(
                "Test dataset not loaded. Call load_test_dataset() first."
            )

        num_classes = len(self.class_names)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_samples = 0
        correct_preds = 0

        # Confusion matrix counters
        true_positives = [0] * num_classes
        false_positives = [0] * num_classes
        false_negatives = [0] * num_classes

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating", leave=True)
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_preds += (preds == labels).sum().item()
                total_samples += images.size(0)

                for cls in range(num_classes):
                    true_positives[cls] += (
                        ((preds == cls) & (labels == cls)).sum().item()
                    )
                    false_positives[cls] += (
                        ((preds == cls) & (labels != cls)).sum().item()
                    )
                    false_negatives[cls] += (
                        ((preds != cls) & (labels == cls)).sum().item()
                    )

        overall_loss = total_loss / total_samples
        overall_accuracy = correct_preds / total_samples

        per_class: list[dict] = []
        for cls in range(num_classes):
            tp = true_positives[cls]
            fp = false_positives[cls]
            fn = false_negatives[cls]
            support = tp + fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class.append(
                {
                    "class_name": self.class_names[cls],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                }
            )

        logging.info(
            f"Evaluation complete — Loss: {overall_loss:.4f}, "
            f"Accuracy: {overall_accuracy:.4f}"
        )

        return {
            "overall_accuracy": overall_accuracy,
            "overall_loss": overall_loss,
            "per_class": per_class,
        }

    # ------------------------------------------------------------------
    # 4. Save results to CSV
    # ------------------------------------------------------------------
    def save_results(self, results: dict[str, object], model_path: str | Path) -> None:
        """Persists evaluation metrics to two CSV files under artifacts/testing_result/.

        Writes:
            - ``per_class_metrics.csv`` — precision, recall, F1, support per class.
            - ``summary.csv`` — overall accuracy and loss for this run.

        Args:
            results: The dictionary returned by :meth:`evaluate`.
            model_path: The checkpoint path used during this evaluation run,
                recorded in the summary for traceability.

        Raises:
            KeyError: If *results* does not contain the expected keys.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # --- per-class CSV ---
        per_class_csv = self.results_dir / "per_class_metrics.csv"
        per_class_data: list[dict] = results["per_class"]  # type: ignore[assignment]
        with open(per_class_csv, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["class_name", "precision", "recall", "f1", "support"],
            )
            writer.writeheader()
            for row in per_class_data:
                writer.writerow(
                    {
                        "class_name": row["class_name"],
                        "precision": f"{row['precision']:.4f}",
                        "recall": f"{row['recall']:.4f}",
                        "f1": f"{row['f1']:.4f}",
                        "support": row["support"],
                    }
                )
        logging.info(f"Per-class metrics saved to {per_class_csv}")

        # --- summary CSV ---
        summary_csv = self.results_dir / "summary.csv"
        summary_exists = summary_csv.exists()
        with open(summary_csv, mode="a", newline="") as f:
            fieldnames = ["model_path", "overall_accuracy", "overall_loss"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not summary_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "model_path": str(model_path),
                    "overall_accuracy": f"{results['overall_accuracy']:.4f}",
                    "overall_loss": f"{results['overall_loss']:.4f}",
                }
            )
        logging.info(f"Summary metrics appended to {summary_csv}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    """Runs the full evaluation pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ResNet-152 model on the test set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    evaluator = ModelEvaluator(config=config)
    evaluator.load_model(model_path=args.model_path)
    evaluator.load_test_dataset()
    results = evaluator.evaluate()
    evaluator.save_results(results=results, model_path=args.model_path)


if __name__ == "__main__":
    main()
