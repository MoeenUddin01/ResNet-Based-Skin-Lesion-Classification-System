from __future__ import annotations

import os
from pathlib import Path

import torch

from src.model.resnet_152 import ResNet152Model

# ---------------------------------------------------------------------------
# Model location resolution
# ---------------------------------------------------------------------------
# Production (HF Spaces): set the env var HF_MODEL_REPO to your repo ID,
#   e.g. "john-doe/skin-lesion-resnet152". The file is downloaded once and
#   cached inside the container.
#
# Development: falls back to the local artifacts/ path.
# ---------------------------------------------------------------------------
_HF_REPO = os.getenv("HF_MODEL_REPO")
_MODEL_FILENAME = "best_model_ep49_acc0.6466_loss1.0435.pth"
_LOCAL_MODEL_PATH = Path("artifacts") / _MODEL_FILENAME


def _resolve_model_path() -> Path:
    """Returns the path to the model weights file.

    Downloads from Hugging Face Hub when the ``HF_MODEL_REPO`` environment
    variable is set; otherwise expects the file at the local artifacts path.

    Returns:
        Path to the model weights file.

    Raises:
        FileNotFoundError: If the model file is not found locally and no HF
            repo is configured.
        RuntimeError: If the Hugging Face Hub download fails.
    """
    if _HF_REPO:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import]

            print(f"Downloading model '{_MODEL_FILENAME}' from HF Hub repo '{_HF_REPO}'...")
            downloaded = hf_hub_download(
                repo_id=_HF_REPO,
                filename=_MODEL_FILENAME,
            )
            return Path(downloaded)
        except Exception as e:
            error_msg = (
                f"Failed to download model from HF Hub repo '{_HF_REPO}'. "
                f"Error: {e}. Ensure HF_MODEL_REPO is correct and the file "
                f"'{_MODEL_FILENAME}' exists in the repository."
            )
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e

    if not _LOCAL_MODEL_PATH.exists():
        error_msg = (
            f"Model file not found at '{_LOCAL_MODEL_PATH.absolute()}'. "
            "To resolve: \n"
            "1. Set 'HF_MODEL_REPO' as an environment variable (repo ID) to download from Hugging Face.\n"
            "2. Ensure the weights file exists locally at the specified path."
        )
        print(f"ERROR: {error_msg}")
        raise FileNotFoundError(error_msg)
    return _LOCAL_MODEL_PATH


def load_inference_model() -> torch.nn.Module:
    """Loads the best trained ResNet-152 model for inference.

    Returns:
        The loaded and eval-ready PyTorch model.

    Raises:
        FileNotFoundError: If the model weights file does not exist.
        RuntimeError: If the model fails to load.
    """
    model_path = _resolve_model_path()

    try:
        model = ResNet152Model(num_classes=7, pretrained=False)
        state_dict = torch.load(model_path, map_location="cpu")

        # Unwrap state dict if it was saved inside a checkpoint dictionary.
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Fallback if only the inner ResNet was saved.
            model.model.load_state_dict(state_dict)

        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from '{model_path}': {e}"
        ) from e

