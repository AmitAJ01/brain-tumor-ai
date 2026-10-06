"""
Dataset Preparation Script for BrainTumorAI
Converts raw .mat files from the Figshare dataset into standardized PNG images,
binary masks, and train/val/test partitions.

Class index convention (as requested by user):
  0 = glioma
  1 = meningioma
  2 = pituitary tumor

Raw .mat mapping:
  cjdata.label == 1 -> meningioma (index 1)
  cjdata.label == 2 -> glioma     (index 0)
  cjdata.label == 3 -> pituitary  (index 2)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Mapping from raw .mat label to project class index & name
RAW_TO_CLASS = {
    1: {"index": 1, "name": "meningioma"},
    2: {"index": 0, "name": "glioma"},
    3: {"index": 2, "name": "pituitary"}
}

def load_mat_data(mat_path: Path):
    """Load cjdata struct from .mat file."""
    try:
        import scipy.io
        mat = scipy.io.loadmat(mat_path)
        cjdata = mat["cjdata"]
        raw_label = int(cjdata["label"][0, 0][0, 0])
        pid = str(cjdata["PID"][0, 0][0])
        image = np.array(cjdata["image"][0, 0], dtype=np.float32)
        tumor_mask = np.array(cjdata["tumorMask"][0, 0], dtype=np.uint8)
        return raw_label, pid, image, tumor_mask
    except Exception:
        import h5py
        with h5py.File(mat_path, "r") as f:
            cjdata = f["cjdata"]
            raw_label = int(cjdata["label"][0, 0])
            pid_raw = cjdata["PID"][:]
            pid = "".join(chr(c[0]) for c in pid_raw)
            image = np.array(cjdata["image"], dtype=np.float32).T
            tumor_mask = np.array(cjdata["tumorMask"], dtype=np.uint8).T
            return raw_label, pid, image, tumor_mask

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize raw MRI float intensities to [0, 255] uint8."""
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        norm = (img - img_min) / (img_max - img_min) * 255.0
    else:
        norm = np.zeros_like(img)
    return norm.astype(np.uint8)

def prepare_dataset(image_size: int = 224, random_state: int = 42):
    """
    Convert all .mat files to organized PNG dataset.
    Splits into train (approx 81%), val (8%), test (11%) adhering to
    the methodology described in Tiwary et al. (2025).
    """
    mat_files = sorted([f for f in RAW_DIR.glob("*.mat") if f.name != "cvind.mat"])
    total_files = len(mat_files)
    if total_files == 0:
        print("[ERROR] No .mat files found in data/raw. Run download_dataset.py first.")
        sys.exit(1)

    print(f"[PREPARE] Processing {total_files} .mat files...")

    # First pass: parse metadata and class labels for stratified split
    metadata = []
    for f in mat_files:
        try:
            raw_label, pid, img, mask = load_mat_data(f)
            cls_info = RAW_TO_CLASS.get(raw_label, {"index": -1, "name": "unknown"})
            metadata.append({
                "file": f.name,
                "path": str(f),
                "raw_label": raw_label,
                "label": cls_info["index"],
                "class_name": cls_info["name"],
                "pid": pid,
            })
        except Exception as e:
            print(f"[WARNING] Skipping unreadable file {f.name}: {e}")

    # Stratified split:
    # 1. Split out validation set (8%)
    train_val_idx, val_idx = train_test_split(
        range(len(metadata)),
        test_size=0.08,
        random_state=random_state,
        stratify=[m["label"] for m in metadata]
    )
    # 2. Split remaining (92%) into train and test (12% of 92% ~ 11.04% of total)
    train_metadata = [metadata[i] for i in train_val_idx]
    train_idx_sub, test_idx_sub = train_test_split(
        range(len(train_metadata)),
        test_size=0.12,
        random_state=random_state,
        stratify=[m["label"] for m in train_metadata]
    )

    actual_train_indices = [train_val_idx[i] for i in train_idx_sub]
    actual_test_indices = [train_val_idx[i] for i in test_idx_sub]
    actual_val_indices = val_idx

    print(f"  Split counts: Train={len(actual_train_indices)}, Val={len(actual_val_indices)}, Test={len(actual_test_indices)}")

    split_map = {}
    for idx in actual_train_indices:
        split_map[metadata[idx]["file"]] = "train"
    for idx in actual_val_indices:
        split_map[metadata[idx]["file"]] = "val"
    for idx in actual_test_indices:
        split_map[metadata[idx]["file"]] = "test"

    # Create target directories
    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / split / "masks").mkdir(parents=True, exist_ok=True)

    manifest = []
    stats = {
        "train": {"glioma": 0, "meningioma": 0, "pituitary": 0},
        "val": {"glioma": 0, "meningioma": 0, "pituitary": 0},
        "test": {"glioma": 0, "meningioma": 0, "pituitary": 0}
    }

    for i, meta in enumerate(metadata):
        if (i + 1) % 500 == 0 or i == total_files - 1:
            print(f"  Writing images {i + 1}/{total_files}...")

        fname = meta["file"]
        split = split_map[fname]
        base_name = Path(fname).stem

        # Load raw data
        raw_label, pid, img, mask = load_mat_data(Path(meta["path"]))
        
        # Normalize and resize
        img_uint8 = normalize_to_uint8(img)
        img_resized = cv2.resize(img_uint8, (image_size, image_size), interpolation=cv2.INTER_AREA)
        
        # Mask is binary 0 or 1 -> convert to 0 or 255 for PNG storage
        mask_binary = (mask > 0).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        # File paths
        img_save_path = PROCESSED_DIR / split / "images" / f"{base_name}.png"
        mask_save_path = PROCESSED_DIR / split / "masks" / f"{base_name}.png"

        # Save as PNG
        cv2.imwrite(str(img_save_path), img_resized)
        cv2.imwrite(str(mask_save_path), mask_resized)

        entry = {
            "id": base_name,
            "split": split,
            "label": meta["label"],
            "class_name": meta["class_name"],
            "patient_id": meta["pid"],
            "image_path": str(img_save_path.relative_to(PROJECT_ROOT)),
            "mask_path": str(mask_save_path.relative_to(PROJECT_ROOT)),
            "tumor_pixel_area": int((mask_resized > 0).sum()),
            "original_shape": list(img.shape),
            "target_shape": [image_size, image_size]
        }
        manifest.append(entry)
        stats[split][meta["class_name"]] += 1

    # Save manifest
    manifest_path = PROCESSED_DIR / "dataset_manifest.json"
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    summary = {
        "total_samples": len(manifest),
        "image_size": [image_size, image_size],
        "splits": {
            "train": len(actual_train_indices),
            "val": len(actual_val_indices),
            "test": len(actual_test_indices)
        },
        "class_breakdown": stats
    }
    with open(PROCESSED_DIR / "dataset_summary.json", "w") as sf:
        json.dump(summary, sf, indent=2)

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)
    print(f"Total processed samples: {len(manifest)}")
    print(f"Train samples: {summary['splits']['train']} (Glioma: {stats['train']['glioma']}, Meningioma: {stats['train']['meningioma']}, Pituitary: {stats['train']['pituitary']})")
    print(f"Val samples:   {summary['splits']['val']} (Glioma: {stats['val']['glioma']}, Meningioma: {stats['val']['meningioma']}, Pituitary: {stats['val']['pituitary']})")
    print(f"Test samples:  {summary['splits']['test']} (Glioma: {stats['test']['glioma']}, Meningioma: {stats['test']['meningioma']}, Pituitary: {stats['test']['pituitary']})")
    print(f"Manifest written to: {manifest_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PNG dataset from raw .mat files")
    parser.add_argument("--size", type=int, default=224, help="Target image size (e.g. 224)")
    args = parser.parse_args()
    prepare_dataset(image_size=args.size)
