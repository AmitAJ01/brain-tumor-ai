"""
Unit tests for Repositories (Repository Pattern)
"""

import pytest
from pathlib import Path
from backend.app.infrastructure.repositories.dataset_repository import DatasetRepository
from backend.app.infrastructure.repositories.model_repository import ModelRepository

def test_dataset_repository():
    repo = DatasetRepository()
    summary = repo.get_summary()
    assert "total_samples" in summary
    assert "splits" in summary

    samples = repo.list_samples(split="test")
    assert isinstance(samples, list)
    if len(samples) > 0:
        sample_id = samples[0]
        data = repo.get_sample_by_id(sample_id)
        assert "image" in data
        assert "mask" in data
        assert "class_name" in data

def test_model_repository():
    repo = ModelRepository()
    path = repo.get_path("brain_tumor_efficientnet.pth")
    assert path.name.endswith(".pth")
