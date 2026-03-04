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

2. **Split Data for Training, Validation and Testing:** Once the processed set is built, run the splitting script to create an official, reproducibe partition of exactly `10%` test data and `10%` validation data. The data will be saved inside `dataset/split/`.

```bash
python -m src.pipeline.split_data
```

## Model Training

With the structured splits inside `dataset/split`, the train and val loaders can be initialized for the training loop. The test data loader is kept strictly for final evaluation metrics.

_(Training commands and logic to be updated later)_

## Serving and App Demo

**Run the Streamlit app:**
```bash
streamlit run app/streamlit.py
```

**Run the FastAPI server:**
```bash
uvicorn app.main:app --reload
```
