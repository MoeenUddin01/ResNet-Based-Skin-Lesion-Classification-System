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
- **Best model weights** (highest validation accuracy) are saved to `artifacts/models/` with the filename pattern `best_model_ep{epoch}_acc{accuracy}_loss{loss}.pth`.

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

## Serving and App Demo

**Run the Streamlit app:**
```bash
streamlit run app/streamlit.py
```

**Run the FastAPI server:**
```bash
uvicorn app.main:app --reload
```
