"""
Unit tests for Model Adapters (Adapter Pattern)
"""

import pytest
import torch
from backend.app.infrastructure.adapters.efficientnet_adapter import EfficientNetAdapter
from backend.app.infrastructure.adapters.efficientnet_unet_adapter import EfficientNetUNetAdapter

def test_efficientnet_adapter_prediction():
    adapter = EfficientNetAdapter(num_classes=3, device="cpu")
    dummy_input = torch.randn(1, 3, 224, 224)
    result = adapter.predict(dummy_input)

    assert "predicted_class" in result
    assert result["predicted_class"] in ["glioma", "meningioma", "pituitary"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert len(result["probabilities"]) == 3
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-2

def test_efficientnet_unet_adapter_prediction():
    adapter = EfficientNetUNetAdapter(num_classes=1, device="cpu")
    dummy_input = torch.randn(1, 3, 224, 224)
    result = adapter.predict(dummy_input)

    assert "tumor_detected" in result
    assert "tumor_area_pixels" in result
    assert "tumor_area_percentage" in result
    assert "mask" in result
    assert result["mask"].shape == (224, 224)
