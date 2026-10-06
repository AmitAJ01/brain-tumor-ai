"""
Unit tests for ModelFactory (Factory Pattern)
"""

import pytest
from backend.app.infrastructure.factories.model_factory import ModelFactory
from backend.app.infrastructure.adapters.efficientnet_adapter import EfficientNetAdapter
from backend.app.infrastructure.adapters.efficientnet_unet_adapter import EfficientNetUNetAdapter

def test_model_factory_creates_classifier():
    adapter = ModelFactory.create("efficientnet", device="cpu")
    assert isinstance(adapter, EfficientNetAdapter)
    info = adapter.get_model_info()
    assert info["architecture"] == "EfficientNet-B0"
    assert info["num_classes"] == 3

def test_model_factory_creates_segmentor():
    adapter = ModelFactory.create("efficient_unet", device="cpu")
    assert isinstance(adapter, EfficientNetUNetAdapter)
    info = adapter.get_model_info()
    assert "U-Net" in info["architecture"]

def test_model_factory_invalid_type():
    with pytest.raises(ValueError):
        ModelFactory.create("invalid_model_type")
