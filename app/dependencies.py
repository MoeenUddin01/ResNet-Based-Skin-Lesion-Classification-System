from functools import lru_cache

import torch

from app.model_loader import load_inference_model


@lru_cache(maxsize=1)
def get_model() -> torch.nn.Module:
    """Provides the trained model as a FastAPI dependency.
    
    Uses lru_cache to ensure the model is only loaded once at startup.

    Returns:
        The loaded PyTorch model.
    """
    return load_inference_model()
