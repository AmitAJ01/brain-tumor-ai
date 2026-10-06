"""
Adapter Pattern Interface for Machine Learning Models
Decouples application layer from PyTorch model internals.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch

class ModelAdapter(ABC):
    """
    Adapter Pattern Interface.
    Provides a unified invocation protocol across diverse models
    (EfficientNet Classifier, EfficientNet-UNet Segmentor, etc.).
    """

    @abstractmethod
    def load_model(self, weights_path: str, device: str = "cpu") -> None:
        """Load trained weights from disk into the encapsulated model."""
        pass

    @abstractmethod
    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Execute forward inference and return normalized prediction dictionary.
        
        Args:
            input_tensor: 4D PyTorch tensor (B, C, H, W).
            
        Returns:
            Dictionary containing model-specific predictions (probabilities, masks, etc.).
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about architecture, parameters, input size, and classes."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if weights have been successfully loaded."""
        pass
