"""
Unit tests for Preprocessing Strategies (Strategy Pattern)
"""

import pytest
import numpy as np
from backend.app.domain.strategies.standard_preprocessing import StandardPreprocessingStrategy
from backend.app.domain.strategies.fuzzy_preprocessing import FuzzyPreprocessingStrategy

def test_standard_preprocessing():
    strategy = StandardPreprocessingStrategy()
    assert strategy.name == "standard"
    assert "Standard Min-Max" in strategy.description

    # Dummy grayscale image
    dummy_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    out = strategy.preprocess(dummy_img, target_size=(224, 224))

    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0

def test_fuzzy_preprocessing():
    strategy = FuzzyPreprocessingStrategy()
    assert strategy.name == "fuzzy"
    assert "Fuzzy" in strategy.description

    # Test with simulated MRI slice having hyperintense tumor region
    dummy_img = np.full((128, 128), 50, dtype=np.uint8)
    dummy_img[40:70, 40:70] = 220 # Bright lesion spot

    out = strategy.preprocess(dummy_img, target_size=(224, 224))

    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
