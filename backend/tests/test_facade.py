"""
Unit tests for BrainTumorAnalysisFacade (Facade Pattern)
"""

import pytest
import numpy as np
from backend.app.application.brain_tumor_facade import BrainTumorAnalysisFacade

def test_facade_analysis_workflow():
    facade = BrainTumorAnalysisFacade()

    # Generate synthetic MRI array
    dummy_mri = np.zeros((224, 224), dtype=np.uint8)
    # Add circular brain region
    yy, xx = np.ogrid[:224, :224]
    circle = (xx - 112) ** 2 + (yy - 112) ** 2 <= 80 ** 2
    dummy_mri[circle] = 120
    # Add tumor lesion
    lesion = (xx - 130) ** 2 + (yy - 100) ** 2 <= 20 ** 2
    dummy_mri[lesion] = 230

    result = facade.analyze(dummy_mri, strategy_name="standard")

    assert "prediction" in result
    assert result["prediction"] in ["glioma", "meningioma", "pituitary"]
    assert "confidence" in result
    assert "probabilities" in result
    assert "gradcam_url" in result
    assert result["gradcam_url"].startswith("data:image/png;base64,")
    assert "processing_time_ms" in result
    assert "disclaimer" in result

def test_facade_strategy_switching():
    facade = BrainTumorAnalysisFacade()
    assert facade.strategy.name == "standard"

    facade.set_preprocessing_strategy("fuzzy")
    assert facade.strategy.name == "fuzzy"

    facade.set_preprocessing_strategy("standard")
    assert facade.strategy.name == "standard"
