"""
Domain Interfaces for Preprocessing Strategies and Model Adapters
Implements Strategy Pattern interface and Adapter Pattern interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class PreprocessingStrategy(ABC):
    """
    Strategy Pattern Interface for Image Preprocessing.
    Allows dynamic substitution of image preprocessing algorithms (e.g. Standard vs Fuzzy)
    without modifying the classification or segmentation pipelines.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the preprocessing algorithm."""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
        """
        Preprocess input MRI image.
        
        Args:
            image: 2D or 3D numpy array representing grayscale or RGB MRI slice.
            target_size: Desired output dimensions (height, width).
            
        Returns:
            Preprocessed 3-channel float32 numpy array with values in [0.0, 1.0].
        """
        pass
