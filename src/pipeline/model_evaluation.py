"""
Model evaluation pipeline for skin lesion classification.

Loads a trained checkpoint, evaluates on the held-out test set, and:
- Saves per-class precision/recall/F1 and a run summary to CSV.
- Logs all metrics, AUC curves (one PNG per class), and CSV artifacts
  to MLflow (backed by DagShub).

Charts produced in DagShub:
  Chart 3  — test accuracy (single scalar per evaluation run)
  Chart 5  — per-class ROC-AUC curves (logged as PNG artifacts)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loader import get_test_loader
from src.model.resnet_152 import ResNet152Model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Loads, evaluates, and persists results for a trained ResNet-152 model.

    Attributes:
        config: Full configuration dictionary loaded from YAML.
        device: The torch device (CPU or CUDA) used for inference.
        model: The ResNet-152 classification model.
        test_loader: DataLoader for the held-out test set.
        class_names: Ordered list of class name strings.
        results_dir: Directory where result CSVs are written.
        auc_dir: Sub-directory inside *results_dir* for AUC curve PNGs.
    """

    def __init__(self, config: dict) -> None:
        """Initialises the evaluator from a configuration dictionary.

        Args:
            config: A dictionary with keys ``hardware``, ``model``,
                ``data``, and ``mlflow``. The ``data`` section must
                contain ``test_dir`` and ``img_size``.
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
        self.auc_dir = self.results_dir / "auc_curves"

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------

    def load_model(self, model_path: str | Path) -> None:
        """Loads a trained ResNet-152 checkpoint from disk.

        Args:
            model_path: Path to the ``.pth`` checkpoint file saved by
                the training pipeline.

        Raises:
            FileNotFoundError: If no checkpoint exists at *model_path*.
            RuntimeError: If the state-dict cannot be loaded into the model.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}"
            )

        model_cfg = self.config["model"]
        num_classes = model_cfg.get("num_classes", 7)
        model = ResNet152Model(num_classes=num_classes, pretrained=False)

        try:
            model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load weights from {path}: {exc}"
            ) from exc

        model.to(self.device)
        model.eval()
        self.model = model
        logging.info(f"Loaded model weights from {path}")

    # ------------------------------------------------------------------
    # 2. Load test dataset
    # ------------------------------------------------------------------

    def load_test_dataset(self) -> None:
        """Loads the test DataLoader using the path in the config.

        Raises:
            KeyError: If ``test_dir`` is missing from the config.
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
            f"Test dataset loaded: "
            f"{len(test_loader.dataset)} samples, "  # type: ignore[arg-type]
            f"classes: {class_names}"
        )

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, object]:
        """Runs inference over the test set and computes all metrics.

        Accumulates raw softmax probabilities to enable per-class OvR AUC
        computation in addition to the standard precision / recall / F1 report.

        Returns:
            A dictionary with keys:
                - ``overall_accuracy`` (float)
                - ``overall_loss`` (float)
                - ``per_class`` (list[dict]) — one dict per class with keys
                  ``class_name``, ``precision``, ``recall``, ``f1``,
                  ``support``, ``auc``.
                - ``all_labels`` (list[int]) — ground-truth label per sample.
                - ``all_probs`` (np.ndarray, shape N×C) — softmax scores.
                - ``macro_auc`` (float) — macro-averaged OvR AUC.

        Raises:
            RuntimeError: If the model or test loader has not been loaded.
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

        true_positives = [0] * num_classes
        false_positives = [0] * num_classes
        false_negatives = [0] * num_classes

        all_labels: list[int] = []
        all_probs_list: list[np.ndarray] = []

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

                # Accumulate raw probabilities for AUC
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs_list.append(probs)
                all_labels.extend(labels.cpu().tolist())

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
        all_probs = np.vstack(all_probs_list)  # shape: (N, C)
        labels_arr = np.array(all_labels)

        # One-hot encode labels for AUC computation
        labels_onehot = np.eye(num_classes)[labels_arr]

        # Macro AUC (handles cases where a class has no positive samples)
        try:
            macro_auc = float(
                roc_auc_score(
                    labels_onehot,
                    all_probs,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError as exc:
            logging.warning(f"Could not compute macro AUC: {exc}")
            macro_auc = float("nan")

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

            # Per-class OvR AUC
            try:
                cls_auc = float(
                    roc_auc_score(labels_onehot[:, cls], all_probs[:, cls])
                )
            except ValueError as exc:
                logging.warning(
                    f"AUC unavailable for class "
                    f"'{self.class_names[cls]}': {exc}"
                )
                cls_auc = float("nan")

            per_class.append(
                {
                    "class_name": self.class_names[cls],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                    "auc": cls_auc,
                }
            )

        logging.info(
            f"Evaluation complete — Loss: {overall_loss:.4f}, "
            f"Accuracy: {overall_accuracy:.4f}, Macro AUC: {macro_auc:.4f}"
        )

        return {
            "overall_accuracy": overall_accuracy,
            "overall_loss": overall_loss,
            "per_class": per_class,
            "all_labels": all_labels,
            "all_probs": all_probs,
            "macro_auc": macro_auc,
        }

    # ------------------------------------------------------------------
    # 4. Save AUC curve PNGs  (Chart 5)
    # ------------------------------------------------------------------

    def save_auc_curves(
        self,
        all_labels: list[int],
        all_probs: np.ndarray,
    ) -> list[Path]:
        """Generates and saves one ROC-AUC PNG per class to disk.

        Files are written to ``artifacts/testing_result/auc_curves/``.

        Args:
            all_labels: Ground-truth integer label for every test sample.
            all_probs: Softmax probability matrix of shape (N, num_classes).

        Returns:
            List of ``Path`` objects pointing to the saved PNG files.
        """
        self.auc_dir.mkdir(parents=True, exist_ok=True)
        num_classes = len(self.class_names)
        labels_arr = np.array(all_labels)
        labels_onehot = np.eye(num_classes)[labels_arr]
        png_paths: list[Path] = []

        for cls, cls_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(labels_onehot[:, cls], all_probs[:, cls])
            try:
                auc_val = roc_auc_score(
                    labels_onehot[:, cls], all_probs[:, cls]
                )
            except ValueError:
                auc_val = float("nan")

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(
                fpr,
                tpr,
                color="#4F81BD",
                lw=2,
                label=f"AUC = {auc_val:.3f}",
            )
            ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1, linestyle="--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title(f"ROC Curve — {cls_name}", fontsize=13)
            ax.legend(loc="lower right", fontsize=11)
            fig.tight_layout()

            png_path = self.auc_dir / f"auc_{cls_name}.png"
            fig.savefig(png_path, dpi=120)
            plt.close(fig)
            png_paths.append(png_path)
            logging.info(f"AUC curve saved: {png_path}")

        return png_paths

    # ------------------------------------------------------------------
    # 5. Save CSV results
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: dict[str, object],
        model_path: str | Path,
    ) -> None:
        """Persists evaluation metrics to two CSV files.

        Writes:
            - ``per_class_metrics.csv`` — precision, recall, F1, AUC,
              and support per class.
            - ``summary.csv`` — overall accuracy and loss per run
              (appended so multiple checkpoints can be compared).

        Args:
            results: The dictionary returned by :meth:`evaluate`.
            model_path: Checkpoint path used for this run (recorded in
                the summary CSV for traceability).

        Raises:
            KeyError: If *results* does not contain the expected keys.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        per_class_csv = self.results_dir / "per_class_metrics.csv"
        per_class_data: list[dict] = results["per_class"]  # type: ignore[assignment]
        with open(per_class_csv, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "class_name",
                    "precision",
                    "recall",
                    "f1",
                    "auc",
                    "support",
                ],
            )
            writer.writeheader()
            for row in per_class_data:
                writer.writerow(
                    {
                        "class_name": row["class_name"],
                        "precision": f"{row['precision']:.4f}",
                        "recall": f"{row['recall']:.4f}",
                        "f1": f"{row['f1']:.4f}",
                        "auc": (
                            f"{row['auc']:.4f}"
                            if not (
                                isinstance(row["auc"], float)
                                and row["auc"] != row["auc"]
                            )
                            else "N/A"
                        ),
                        "support": row["support"],
                    }
                )
        logging.info(f"Per-class metrics saved to {per_class_csv}")

        summary_csv = self.results_dir / "summary.csv"
        summary_exists = summary_csv.exists()
        with open(summary_csv, mode="a", newline="") as f:
            fieldnames = [
                "model_path",
                "overall_accuracy",
                "overall_loss",
                "macro_auc",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not summary_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "model_path": str(model_path),
                    "overall_accuracy": (
                        f"{results['overall_accuracy']:.4f}"  # type: ignore[arg-type]
                    ),
                    "overall_loss": (
                        f"{results['overall_loss']:.4f}"  # type: ignore[arg-type]
                    ),
                    "macro_auc": (
                        f"{results['macro_auc']:.4f}"  # type: ignore[arg-type]
                    ),
                }
            )
        logging.info(f"Summary metrics appended to {summary_csv}")

    # ------------------------------------------------------------------
    # 6. MLflow-wrapped evaluation  (main public entry-point)
    # ------------------------------------------------------------------

    def run_mlflow_evaluation(self, model_path: str | Path) -> None:
        """Runs the full evaluation inside an MLflow run.

        Sets the tracking URI and experiment from the config, then inside
        a single :func:`mlflow.start_run` context:

        1. Calls :meth:`evaluate` to compute all metrics.
        2. Calls :meth:`save_results` to write CSVs.
        3. Calls :meth:`save_auc_curves` to save per-class ROC PNGs.
        4. Logs all scalars, CSVs, and PNG files to MLflow/DagShub.

        Args:
            model_path: Path to the ``.pth`` checkpoint to evaluate.

        Raises:
            KeyError: If the ``mlflow`` section is absent from the config.
        """
        load_dotenv()
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if not tracking_uri:
            raise EnvironmentError(
                "MLFLOW_TRACKING_URI is not set. Add it to your .env file."
            )
        mlflow_cfg = self.config["mlflow"]
        date_suffix = datetime.now().strftime("%d_%m_%Y")
        experiment_name = f"{mlflow_cfg['experiment_name']}_{date_suffix}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logging.info(
            f"MLflow tracking URI: {tracking_uri} | "
            f"Experiment: {experiment_name}"
        )

        with mlflow.start_run(run_name="evaluation"):
            mlflow.set_tag("checkpoint", str(model_path))
            mlflow.set_tag("pipeline", "evaluation")

            # ----------------------------------------------------------
            # Run evaluation
            # ----------------------------------------------------------
            results = self.evaluate()

            # ----------------------------------------------------------
            # Scalar metrics  (Chart 3: test accuracy)
            # ----------------------------------------------------------
            mlflow.log_metric(
                "test_accuracy", float(results["overall_accuracy"])  # type: ignore[arg-type]
            )
            mlflow.log_metric(
                "test_loss", float(results["overall_loss"])  # type: ignore[arg-type]
            )
            mlflow.log_metric(
                "macro_auc", float(results["macro_auc"])  # type: ignore[arg-type]
            )

            per_class_data: list[dict] = results["per_class"]  # type: ignore[assignment]
            for row in per_class_data:
                cls_name = row["class_name"]
                mlflow.log_metric(f"{cls_name}_precision", row["precision"])
                mlflow.log_metric(f"{cls_name}_recall", row["recall"])
                mlflow.log_metric(f"{cls_name}_f1", row["f1"])
                if not (
                    isinstance(row["auc"], float) and row["auc"] != row["auc"]
                ):
                    mlflow.log_metric(f"{cls_name}_auc", row["auc"])

            # ----------------------------------------------------------
            # CSV artifacts
            # ----------------------------------------------------------
            self.save_results(results, model_path)
            mlflow.log_artifact(
                str(self.results_dir / "per_class_metrics.csv"),
                artifact_path="evaluation",
            )
            mlflow.log_artifact(
                str(self.results_dir / "summary.csv"),
                artifact_path="evaluation",
            )

            # ----------------------------------------------------------
            # AUC curves  (Chart 5)
            # ----------------------------------------------------------
            png_paths = self.save_auc_curves(
                all_labels=results["all_labels"],  # type: ignore[arg-type]
                all_probs=results["all_probs"],  # type: ignore[arg-type]
            )
            for png_path in png_paths:
                mlflow.log_artifact(
                    str(png_path), artifact_path="auc_curves"
                )

            logging.info(
                "MLflow evaluation run complete. "
                f"Test accuracy: {results['overall_accuracy']:.4f}, "
                f"Macro AUC: {results['macro_auc']:.4f}"
            )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Runs the full MLflow-tracked evaluation pipeline from the command line."""
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
    evaluator.run_mlflow_evaluation(model_path=args.model_path)


if __name__ == "__main__":
    main()
