"""
EfficientNet-UNet Segmentation Model Adapter
Implements Adapter Pattern to adapt EfficientNet-UNet segmentation model to ModelAdapter interface.
"""

from typing import Dict, Any, Optional
import os
import torch
import numpy as np
from pathlib import Path

from backend.app.domain.interfaces.model_adapter import ModelAdapter
from backend.app.infrastructure.models.efficientnet_unet import build_efficientnet_unet

class EfficientNetUNetAdapter(ModelAdapter):
    """
    Adapter Pattern Implementation for EfficientNet-UNet Segmentation Model.
    Decouples raw segmentation tensor manipulation and produces normalized mask
    and tumor area metrics.
    """

    def __init__(self, num_classes: int = 1, device: str = "cpu"):
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.model = build_efficientnet_unet(num_classes=num_classes, pretrained=False)
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = False
        self.weights_path: Optional[str] = None

    def load_model(self, weights_path: str, device: Optional[str] = None) -> None:
        """Load trained segmentation weights."""
        if device is not None:
            self.device = torch.device(device)
            self.model.to(self.device)

        weights_file = Path(weights_path)
        if not weights_file.exists():
            raise FileNotFoundError(f"Segmentation weights not found: {weights_path}")

        checkpoint = torch.load(weights_file, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._is_loaded = True
        self.weights_path = str(weights_file)

    def predict(self, input_tensor: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Execute segmentation inference.
        
        Args:
            input_tensor: 4D Tensor (B, 3, H, W)
            threshold: Probability threshold for binarization (default 0.5)
            
        Returns:
            Dictionary containing mask array, probability map, and tumor area metrics.
        """
        self.model.eval()
        tensor_dev = input_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_dev)
            probs = torch.sigmoid(logits)[0, 0] # (H, W)
            prob_map = probs.cpu().numpy()
            binary_mask = (prob_map >= threshold).astype(np.uint8) * 255

        # Calculate estimated tumor area
        tumor_pixels = int(np.sum(binary_mask > 0))
        
        # Calculate non-background brain parenchyma area from input tensor
        slice_2d = tensor_dev[0].mean(dim=0).cpu().numpy()
        min_val, max_val = float(slice_2d.min()), float(slice_2d.max())
        bg_thresh = min_val + 0.15 * max(max_val - min_val, 1e-4)
        brain_pixels = int(np.sum(slice_2d > bg_thresh))
        if brain_pixels == 0:
            brain_pixels = slice_2d.size

        # Clamp tumor area to non-background brain area (0.0% to 100.0%)
        ratio = min(1.0, tumor_pixels / max(brain_pixels, 1))
        tumor_area_percentage = round(ratio * 100.0, 2)

        return {
            "model_type": "EfficientNet-UNet",
            "tumor_detected": bool(tumor_pixels > 20),
            "tumor_area_pixels": tumor_pixels,
            "total_brain_pixels": brain_pixels,
            "tumor_area_percentage": tumor_area_percentage,
            "area_estimate_note": "Estimated 2D area from segmentation mask. Not clinical tumor volume.",
            "mask": binary_mask,
            "probability_map": prob_map
        }

    def get_model_info(self) -> Dict[str, Any]:
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "name": "EfficientNet-UNet Brain Tumor Segmentor",
            "architecture": "EfficientNet-B0 + U-Net Decoder",
            "num_classes": self.num_classes,
            "total_parameters": param_count,
            "trainable_parameters": trainable_count,
            "is_loaded": self._is_loaded,
            "weights_path": self.weights_path,
            "device": str(self.device),
            "input_resolution": [224, 224, 3]
        }

    def is_loaded(self) -> bool:
        return self._is_loaded
