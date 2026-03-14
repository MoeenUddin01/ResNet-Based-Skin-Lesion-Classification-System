from pathlib import Path

import torch

from src.model.resnet_152 import ResNet152Model

# Hardcoded model path provided by the user
MODEL_PATH = Path("artifacts/best_model_ep49_acc0.6466_loss1.0435.pth")


def load_inference_model() -> torch.nn.Module:
    """Loads the best trained ResNet-152 model for inference.

    Returns:
        The loaded and eval-ready PyTorch model.

    Raises:
        FileNotFoundError: If the model weights file does not exist.
        RuntimeError: If the model fails to load.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    try:
        # We know HAM10000 has 7 classes
        model = ResNet152Model(num_classes=7, pretrained=False)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        
        # Unwrap state dict if it was wrapped in a checkpoint dictionary
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Fallback in case just the inner ResNet was saved
            model.model.load_state_dict(state_dict)

        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}") from e
