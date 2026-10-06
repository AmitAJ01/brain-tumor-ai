"""
Model Repository Implementation
Implements Repository Pattern to manage saving, loading, and querying
deep learning model checkpoints and their accompanying evaluation metrics.
"""

from typing import Dict, Any, Optional
import os
import json
import torch
from pathlib import Path

from backend.app.domain.interfaces.repositories import IModelRepository

class ModelRepository(IModelRepository):
    """
    Concrete Repository for model artifacts.
    Standardizes checkpoint persistence and retrieval.
    """

    def __init__(self, models_root: Optional[Path] = None):
        if models_root is None:
            self.models_root = Path(__file__).resolve().parent.parent.parent.parent.parent / "models"
        else:
            self.models_root = Path(models_root)

        self.models_root.mkdir(parents=True, exist_ok=True)
        (self.models_root / "classifier").mkdir(parents=True, exist_ok=True)
        (self.models_root / "segmentation").mkdir(parents=True, exist_ok=True)

    def _resolve_model_path(self, model_name: str) -> Path:
        """Resolve path whether provided as bare filename, relative or subfolder."""
        name = Path(model_name).name
        if not name.endswith(".pth"):
            name = f"{name}.pth"

        # Check direct path
        direct = self.models_root / name
        if direct.exists():
            return direct
        # Check subfolders
        for sub in ["classifier", "segmentation"]:
            sub_path = self.models_root / sub / name
            if sub_path.exists():
                return sub_path

        # If saving new file, route by keyword
        if "unet" in name.lower() or "seg" in name.lower():
            return self.models_root / "segmentation" / name
        return self.models_root / "classifier" / name

    def save_checkpoint(self, model_name: str, state_dict: dict, metrics: Optional[Dict[str, Any]] = None) -> str:
        dest_path = self._resolve_model_path(model_name)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_name": model_name,
            "state_dict": state_dict,
            "metrics": metrics or {}
        }
        torch.save(payload, dest_path)

        # Save sidecar metadata JSON
        meta_path = dest_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metrics or {}, f, indent=2)

        return str(dest_path)

    def load_checkpoint(self, model_name: str) -> Dict[str, Any]:
        dest_path = self._resolve_model_path(model_name)
        if not dest_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {dest_path}")
        checkpoint = torch.load(dest_path, map_location="cpu")
        return checkpoint

    def exists(self, model_name: str) -> bool:
        dest_path = self._resolve_model_path(model_name)
        return dest_path.exists()

    def get_path(self, model_name: str) -> Path:
        return self._resolve_model_path(model_name)
