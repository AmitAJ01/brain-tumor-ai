"""
Standard Image Preprocessing Strategy
Implements standard MRI normalization, resizing, and 3-channel conversion.
"""

import numpy as np
import cv2
from PIL import Image
from backend.app.domain.interfaces.preprocessing_strategy import PreprocessingStrategy

class StandardPreprocessingStrategy(PreprocessingStrategy):
    """
    Standard preprocessing pipeline:
    1. Input format normalization (PIL / NumPy array to uint8/float32)
    2. Resize to target dimension (default 224x224)
    3. Min-Max normalization to [0.0, 1.0]
    4. Grayscale to 3-channel replication for ImageNet-pretrained CNN backbones
    """

    @property
    def name(self) -> str:
        return "standard"

    @property
    def description(self) -> str:
        return "Standard Min-Max normalization with bicubic interpolation and 3-channel replication."

    def preprocess(self, image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Handle float or 16-bit input
        image = image.astype(np.float32)
        
        # Min-max normalization of raw intensities
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)

        # Ensure single channel before resizing
        if image.ndim == 3 and image.shape[2] == 3:
            # If already 3-channel, convert to grayscale first to standardize
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[:, :, 0]
        else:
            gray = image

        # Resize to target size (height, width)
        resized = cv2.resize(gray, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)
        resized = np.clip(resized, 0.0, 1.0).astype(np.float32)

        # Stack to 3 channels (H, W, 3)
        three_channel = np.stack([resized, resized, resized], axis=-1)
        return three_channel
