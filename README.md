# ResNet-Based Skin Lesion Classification System

Skin lesion image classification using a ResNet-152 backbone (PyTorch/torchvision). The codebase follows a modular ML pipeline pattern with separate concerns for data, model, and pipeline orchestration.

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

## Data Preparation

The `dataset/` directory contains all data for the application. The workflow for organizing the data prior to training is as follows:

1. **Organize Raw Data:** Start by combining the raw HAM10000 `part_1` and `part_2` datasets, then organize the images into class-specific subdirectories inside `dataset/processed/`.

```bash
python -m src.data.datapre
```

2. **Split Data for Training, Validation and Testing:** Once the processed set is built, run the splitting script to create an official, reproducible partition of exactly `10%` test data. The remaining data is then split into `30%` validation data and `70%` training data. The data will be saved inside `dataset/split/`.

```bash
python -m src.pipeline.split_data
```

## Model Training

The training pipeline uses a custom ResNet-152 wrapper that handles class weights and dynamic augmentation for minority classes.

### Short Guide: Training Your Model

1. **Configure Hyperparameters:** Open `config.yaml` to adjust your settings before training. This file controls:
   - **hardware**: Enabling CUDA and setting dataloader workers.
   - **model**: Toggling pretrained ImageNet weights and freezing the backbone.
   - **data**: Selecting the dataset directory (`split_dir`), test data directory (`test_dir`), and image sizes.
   - **`subset_fraction`** (dev only): Available in `config.dev.yaml`, set to e.g. `0.1` to train on only 10% of data for quick local debugging.

2. **Run the Pipeline:** With your data split and configurations set, start the training orchestrator:

   - **For Full Training:** (Uses `config.yaml` by default)
     ```bash
     uv run python -m src.pipeline.model_training
     ```

   - **For Local Development/Testing:** (Uses the faster `config.dev.yaml`)
     ```bash
     uv run python -m src.pipeline.model_training --config config.dev.yaml
     ```

The script will automatically detect device capabilities, apply dataset subsetting and weighted cross-entropy loss based on class imbalances, display dynamic progress bars, and log epoch metrics.

Once training finishes:
- **Best model checkpointing**: The script continuously monitors your model's validation accuracy at the end of every epoch. The weights that achieve the absolute highest `val_acc` during the run are automatically saved as `best_model.pth` to ensure the optimal model is retained (preventing overfitting).
- The `best_model.pth` file is also automatically uploaded directly to your **DagShub MLflow Artifacts** store so it is backed up in the cloud.

## Model Evaluation

After training, evaluate your model on the held-out test set using the evaluation pipeline.

1. **Ensure `test_dir` is set** in your config file (it points to the directory containing the `test/` subfolder):
   ```yaml
   data:
     test_dir: "dataset/split"
   ```

2. **Run the evaluator**, passing the path to your saved checkpoint:
   ```bash
   # Full evaluation
   uv run python -m src.pipeline.model_evaluation \
     --config config.yaml \
     --model-path artifacts/models/<your_checkpoint>.pth

   # Quick evaluation with dev config
   uv run python -m src.pipeline.model_evaluation \
     --config config.dev.yaml \
     --model-path artifacts/models/<your_checkpoint>.pth
   ```

The evaluator automatically:
- Loads the model checkpoint.
- Runs inference over the entire test set.
- Computes overall accuracy, loss, and per-class precision, recall, and F1-score.
- Saves two CSV files to `artifacts/testing_result/`:

| File | Contents |
|---|---|
| `per_class_metrics.csv` | Precision, recall, F1, and support for each of the 7 lesion classes |
| `summary.csv` | Overall accuracy and loss per run (appended so you can compare multiple checkpoints) |

## MLflow Experiment Tracking (DagShub)

Both the training and evaluation pipelines log metrics, params, and artifacts to **MLflow** backed by **DagShub**.

### One-time Setup

1. **Fill in your DagShub repo URL** in both `config.yaml` and `config.dev.yaml`:
   ```yaml
   mlflow:
     tracking_uri: "https://dagshub.com/<your-username>/<your-repo>.mlflow"
     experiment_name: "skin-lesion-resnet152"
   ```

2. **Set authentication** via environment variables (or add to `.env`):
   ```bash
   export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
   export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
   ```
   Get your token from **DagShub → Settings → Tokens**.

### What Gets Logged

| Pipeline | Metrics | Artifacts |
|---|---|---|
| **Training** | `train_loss`, `train_acc`, `val_loss`, `val_acc` per epoch; `best_val_acc`, `best_val_loss` | `best_model.pth` (uploaded automatically based on highest val_acc per-epoch) |
| **Evaluation** | `test_accuracy`, `test_loss`, `macro_auc`; per-class precision / recall / F1 / AUC | `per_class_metrics.csv`, `summary.csv`, 7 AUC curve PNGs |

### DagShub Charts

| # | Chart | How it appears |
|---|---|---|
| 1 | Training accuracy vs epochs | `train_acc` metric (step = epoch) |
| 2 | Validation accuracy vs epochs | `val_acc` metric (step = epoch) |
| 3 | Test accuracy | `test_accuracy` scalar (single point per evaluation run) |
| 4 | Train loss vs val loss vs epochs | `train_loss` + `val_loss` metrics (step = epoch) |
| 5 | ROC-AUC curves | PNG artifacts in `auc_curves/` folder |

AUC curve PNGs are also saved locally to `artifacts/testing_result/auc_curves/`.

---

## Training on Kaggle

Use `config.kaggle.yaml` to train on Kaggle GPU with the HAM10000 dataset mounted from Kaggle directly. Run these cells in order in your Kaggle notebook:

**Cell 1 — Clone repo & install dependencies:**
```bash
!git clone https://github.com/MoeenUddin01/ResNet-Based-Skin-Lesion-Classification-System.git /kaggle/working/project
%cd /kaggle/working/project
!pip install -q mlflow scikit-learn python-dotenv pyyaml tqdm
```

**Cell 2 — Set MLflow credentials from Kaggle Secrets:**
```python
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["MLFLOW_TRACKING_URI"]      = secrets.get_secret("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = secrets.get_secret("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = secrets.get_secret("MLFLOW_TRACKING_PASSWORD")
```

**Cell 3 — Stage data to writable directory:**
```python
import shutil
from pathlib import Path
KAGGLE_INPUT = Path("/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000")
RAW_DIR = Path("/kaggle/working/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
for src, dst in [("HAM10000_images_part_1", "ham10000_images_part_1"),
                 ("HAM10000_images_part_2", "ham10000_images_part_2")]:
    if not (RAW_DIR / dst).exists():
        shutil.copytree(str(KAGGLE_INPUT / src), str(RAW_DIR / dst))
shutil.copy(str(KAGGLE_INPUT / "HAM10000_metadata.csv"), str(RAW_DIR / "HAM10000_metadata.csv"))
```

**Cell 4 — Organise, split, train, and evaluate:**
```bash
!python -m src.data.datapre \
  --raw-dir /kaggle/working/raw \
  --processed-dir /kaggle/working/processed

!python -m src.pipeline.split_data \
  --processed-dir /kaggle/working/processed \
  --output-dir /kaggle/working/split

!python -m src.pipeline.model_training --config config.kaggle.yaml

!python -m src.pipeline.model_evaluation \
  --config config.kaggle.yaml \
  --model-path artifacts/models/<your-checkpoint>.pth
```

> **Note:** Add `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` as Kaggle Secrets before running (notebook sidebar → Add-ons → Secrets).

---

## Serving and App Demo

**Run the Streamlit app:**
```bash
streamlit run app/streamlit.py
```

**Run the FastAPI server:**
```bash
uvicorn app.main:app --reload
```
