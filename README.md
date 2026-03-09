# ResNet-Based Skin Lesion Classification System

Skin lesion image classification using a ResNet-152 backbone (PyTorch/torchvision). The codebase follows a modular ML pipeline pattern with separate concerns for data, model, and pipeline orchestration. 

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

## Data Preparation

The `dataset/` directory contains all data for the application. The workflow for organizing the data prior to training is as follows:

1. **Organize Raw Data:** Start by combining the raw HAM10000 `part_1` and `part_2` datasets, then organize the images into class-specific subdirectories inside `dataset/processed/`.

```bash
python -m src.data.datapre
```

2. **Split Data for Training, Validation and Testing:** Once the processed set is built, run the splitting script to create an official, reproducibe partition of exactly `10%` test data. The remaining data is then split into `30%` validation data and `70%` training data. The data will be saved inside `dataset/split/`.

```bash
python -m src.pipeline.split_data
```

## Model Training

The training pipeline uses a custom ResNet-152 wrapper that handles class weights and dynamic augmentation for minority classes.

### Short Guide: Training Your Model

1. **Configure Hyperparameters:** Open `config.yaml` to adjust your settings before training. This file controls:
   - **hardware**: Enabling CUDA and setting dataloader workers.
   - **model**: Toggling pretrained ImageNet weights and freezing the backbone.
   - **data**: Selecting the dataset directory, image sizes, and `subset_fraction` (e.g., `0.1` to train extremely fast on only 10% of the data for local debugging).

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

Once training finishes, your weights with the highest validation accuracy will automatically be saved into the `artifacts/models/` directory!

## Serving and App Demo

**Run the Streamlit app:**
```bash
streamlit run app/streamlit.py
```

**Run the FastAPI server:**
```bash
uvicorn app.main:app --reload
```
