"""
EfficientNet Model Adapter
Implements Adapter Pattern to adapt EfficientNet-B0 classifier to ModelAdapter interface.
"""

from typing import Dict, Any, Optional
import os
import torch
import torch.nn.functional as F
from pathlib import Path

from backend.app.domain.interfaces.model_adapter import ModelAdapter
from backend.app.infrastructure.models.efficientnet_model import (
    build_efficientnet_b0,
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS
)

class EfficientNetAdapter(ModelAdapter):
    """
    Adapter Pattern Implementation for EfficientNet-B0 Classifier.
    Decouples raw PyTorch model operations behind a standardized application contract.
    """

    def __init__(self, num_classes: int = 3, device: str = "cpu"):
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.model = build_efficientnet_b0(num_classes=num_classes, pretrained=False)
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = False
        self.weights_path: Optional[str] = None

    def load_model(self, weights_path: str, device: Optional[str] = None) -> None:
        """Load checkpoint weights from disk."""
        if device is not None:
            self.device = torch.device(device)
            self.model.to(self.device)

        weights_file = Path(weights_path)
        if not weights_file.exists():
            raise FileNotFoundError(f"Model weights file not found: {weights_path}")

        checkpoint = torch.load(weights_file, map_location=self.device)
        # Check if wrapped in checkpoint dictionary
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

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Execute classification inference.
        
        Args:
            input_tensor: 4D Tensor (B, 3, H, W).
            
        Returns:
            Structured prediction dictionary.
        """
        if not self._is_loaded:
            # Fallback or warning
            pass

        self.model.eval()
        tensor_dev = input_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_dev)
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())

        prob_dict = {
            CLASS_NAMES[i]: round(float(probs[i].item()), 4)
            for i in range(self.num_classes)
        }

        return {
            "model_type": "EfficientNet-B0",
            "class_index": pred_idx,
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100.0, 2),
            "probabilities": prob_dict,
            "logits": [round(float(x), 4) for x in logits[0].cpu().numpy().tolist()]
        }

    def get_model_info(self) -> Dict[str, Any]:
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "name": "EfficientNet-B0 Brain Tumor Classifier",
            "architecture": "EfficientNet-B0",
            "num_classes": self.num_classes,
            "classes": CLASS_NAMES,
            "total_parameters": param_count,
            "trainable_parameters": trainable_count,
            "is_loaded": self._is_loaded,
            "weights_path": self.weights_path,
            "device": str(self.device),
            "input_resolution": [224, 224, 3]
        }

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_target_layer_for_gradcam(self):
        """Return the target convolutional layer for Grad-CAM explainability."""
        # Last MBConv / ConvNormActivation block of features
        return self.model.features[-1]
