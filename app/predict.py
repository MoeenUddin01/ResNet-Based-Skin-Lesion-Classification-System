import io

import torch
from PIL import Image
from torchvision import transforms

from app.schemas import PredictionResponse

# Sorted alphabetically based on dataset/split/train directory structure
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocesses raw image bytes into a tensor for the model.

    Args:
        image_bytes: Raw bytes of the image file.

    Returns:
        A preprocessed image tensor of shape (1, 3, 224, 224).

    Raises:
        ValueError: If the image cannot be opened or processed.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image format: {e}") from e

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension


def predict_image(image_bytes: bytes, model: torch.nn.Module) -> PredictionResponse:
    """Runs inference on an uploaded image.

    Args:
        image_bytes: Raw bytes of the image file.
        model: The trained PyTorch model.

    Returns:
        A PredictionResponse object containing the predicted class and confidences.
    """
    input_tensor = preprocess_image(image_bytes)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    probs_dict = {
        CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)
    }
    
    max_prob, predicted_idx = torch.max(probabilities, dim=0)
    best_class = CLASS_NAMES[predicted_idx.item()]
    
    return PredictionResponse(
        class_name=best_class,
        confidence=float(max_prob),
        probabilities=probs_dict,
    )
