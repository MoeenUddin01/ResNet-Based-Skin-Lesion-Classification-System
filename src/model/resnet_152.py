"""
ResNet-152 model wrapper for skin lesion classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet152_Weights, resnet152


class ResNet152Model(nn.Module):
    """ResNet-152 wrapper for Skin Lesion Classification.
    
    Provides methods for loading pretrained weights, freezing layers, 
    and modifying the classification head for fine-tuning.
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True) -> None:
        """Initializes the ResNet-152 model.

        Args:
            num_classes: The number of output classes.
            pretrained: Whether to load ImageNet pretrained weights.
        """
        super().__init__()
        self.num_classes = num_classes
        self.model = self._load_pretrained(pretrained)
        self.finetune()

    def _load_pretrained(self, pretrained: bool) -> torch.nn.Module:
        """Loads the ResNet-152 model.

        Args:
            pretrained: If True, loads the default recommended pretrained weights.

        Returns:
            The ResNet-152 model.
        """
        weights = ResNet152_Weights.DEFAULT if pretrained else None
        return resnet152(weights=weights)

    def finetune(self) -> None:
        """Modifies the final fully connected layer for fine-tuning.
        
        Replaces the internal 1000-class ImageNet head with a new linear 
        layer matching the target number of classes.
        """
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)

    def freeze_layers(self) -> None:
        """Freezes all layers except the fully connected classification head.
        
        This prevents updates to the feature extraction backbone during training.
        """
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.model(x)
