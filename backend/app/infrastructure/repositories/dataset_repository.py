"""
Dataset Repository Implementation
Implements Repository Pattern to encapsulate data access, partitioning,
and manifest retrieval for the Brain Tumor MRI dataset.
"""

from typing import Dict, Any, List, Optional
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image

from backend.app.domain.interfaces.repositories import IDatasetRepository

class DatasetRepository(IDatasetRepository):
    """
    Concrete Repository for accessing processed brain tumor MRI data.
    Provides structured querying of dataset samples, ground truth masks, and metadata.
    """

    def __init__(self, data_root: Optional[Path] = None):
        if data_root is None:
            # Default to <project_root>/data
            self.data_root = Path(__file__).resolve().parent.parent.parent.parent.parent / "data"
        else:
            self.data_root = Path(data_root)

        self.processed_dir = self.data_root / "processed"
        self.raw_dir = self.data_root / "raw"
        self.manifest_path = self.processed_dir / "dataset_manifest.json"
        self.summary_path = self.processed_dir / "dataset_summary.json"
        self._manifest_cache: Optional[List[Dict[str, Any]]] = None

    def _get_manifest(self) -> List[Dict[str, Any]]:
        if self._manifest_cache is None:
            if not self.manifest_path.exists():
                return []
            with open(self.manifest_path, "r") as f:
                self._manifest_cache = json.load(f)
        return self._manifest_cache

    def get_summary(self) -> Dict[str, Any]:
        """Fetch precomputed dataset summary or build from manifest."""
        if self.summary_path.exists():
            with open(self.summary_path, "r") as f:
                return json.load(f)

        manifest = self._get_manifest()
        splits = {"train": 0, "val": 0, "test": 0}
        classes = {}
        for item in manifest:
            sp = item.get("split", "unknown")
            splits[sp] = splits.get(sp, 0) + 1
            c = item.get("class_name", "unknown")
            classes[c] = classes.get(c, 0) + 1

        return {
            "total_samples": len(manifest),
            "splits": splits,
            "classes": classes
        }

    def list_samples(self, split: Optional[str] = None) -> List[str]:
        manifest = self._get_manifest()
        if split is not None:
            return [m["id"] for m in manifest if m.get("split") == split]
        return [m["id"] for m in manifest]

    def get_sample_by_id(self, sample_id: str) -> Dict[str, Any]:
        manifest = self._get_manifest()
        matching = [m for m in manifest if m["id"] == str(sample_id)]
        if not matching:
            raise KeyError(f"Sample with ID {sample_id} not found in dataset manifest.")

        item = matching[0]
        root = self.data_root.parent
        img_path = root / item["image_path"]
        mask_path = root / item["mask_path"]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        return {
            "id": item["id"],
            "split": item.get("split"),
            "label": item.get("label"),
            "class_name": item.get("class_name"),
            "patient_id": item.get("patient_id"),
            "image": np.array(img),
            "mask": np.array(mask),
            "tumor_pixel_area": item.get("tumor_pixel_area"),
            "image_path": str(img_path),
            "mask_path": str(mask_path)
        }
