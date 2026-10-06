"""
Dataset Download and Verification Script for BrainTumorAI
Downloads and verifies the Figshare Brain Tumor Dataset (Cheng et al., 2017)
Referenced by Tiwary et al. (2025), Sultan et al. (2019), and Almufareh et al. (2024).

Dataset details:
- 3064 T1-weighted contrast-enhanced MRI images
- 233 patients
- 3 classes: meningioma (708), glioma (1426), pituitary (930)
- Struct fields: cjdata.label, cjdata.PID, cjdata.image, cjdata.tumorBorder, cjdata.tumorMask
"""

import os
import sys
import json
import zipfile
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_SUMMARY_PATH = RAW_DATA_DIR / "dataset_summary.json"

FIGSHARE_FILES = [
    {
        "name": "brainTumorDataPublic_1-766.zip",
        "part": 1,
        "slices": (1, 766),
        "size": 214401279,
        "url": "https://ndownloader.figshare.com/files/3381290",
    },
    {
        "name": "brainTumorDataPublic_767-1532.zip",
        "part": 2,
        "slices": (767, 1532),
        "size": 217848429,
        "url": "https://ndownloader.figshare.com/files/3381296",
    },
    {
        "name": "brainTumorDataPublic_1533-2298.zip",
        "part": 3,
        "slices": (1533, 2298),
        "size": 215563856,
        "url": "https://ndownloader.figshare.com/files/3381293",
    },
    {
        "name": "brainTumorDataPublic_2299-3064.zip",
        "part": 4,
        "slices": (2299, 3064),
        "size": 231679762,
        "url": "https://ndownloader.figshare.com/files/3381302",
    },
    {
        "name": "cvind.mat",
        "part": 0,
        "slices": None,
        "size": 5736,
        "url": "https://ndownloader.figshare.com/files/7005344",
    },
]

# Label mapping according to the dataset documentation:
# 1 = meningioma, 2 = glioma, 3 = pituitary
LABEL_MAP_RAW = {1: "meningioma", 2: "glioma", 3: "pituitary"}

def download_file(url: str, dest_path: Path, expected_size: Optional[int] = None) -> bool:
    """Download a file with resume support and progress output."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    
    initial_bytes = temp_path.stat().st_size if temp_path.exists() else 0
    if dest_path.exists():
        if expected_size and dest_path.stat().st_size == expected_size:
            print(f"[EXISTS] {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")
            return True
        elif not expected_size and dest_path.stat().st_size > 0:
            print(f"[EXISTS] {dest_path.name}")
            return True

    headers = {}
    if initial_bytes > 0:
        headers["Range"] = f"bytes={initial_bytes}-"
        print(f"[RESUME] Resuming {dest_path.name} from byte {initial_bytes}...")
    else:
        print(f"[DOWNLOAD] Downloading {dest_path.name}...")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        mode = "ab" if initial_bytes > 0 and response.status_code == 206 else "wb"
        if mode == "wb":
            initial_bytes = 0

        total_size = int(response.headers.get("content-length", 0)) + initial_bytes
        downloaded = initial_bytes

        with open(temp_path, mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  -> Progress: {mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%)")
                    else:
                        sys.stdout.write(f"\r  -> Downloaded {downloaded / (1024*1024):.1f} MB")
                    sys.stdout.flush()
        print()

        if temp_path.exists():
            temp_path.replace(dest_path)
            print(f"[SUCCESS] Saved to {dest_path.name}")
            return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {dest_path.name}: {e}")
        return False
    return False

def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract zip file to directory."""
    print(f"[EXTRACT] Extracting {zip_path.name} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        print(f"[SUCCESS] Extracted {zip_path.name}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to extract {zip_path.name}: {e}")
        return False

def load_single_mat(filepath: Path) -> Dict[str, Any]:
    """Robustly load a .mat file using scipy.io or h5py."""
    try:
        import scipy.io
        mat = scipy.io.loadmat(filepath)
        cjdata = mat["cjdata"]
        label = int(cjdata["label"][0, 0][0, 0])
        pid = str(cjdata["PID"][0, 0][0])
        image = np.array(cjdata["image"][0, 0], dtype=np.float32)
        tumor_mask = np.array(cjdata["tumorMask"][0, 0], dtype=np.uint8)
        tumor_border = cjdata["tumorBorder"][0, 0] if "tumorBorder" in cjdata.dtype.names else None
        return {
            "label": label,
            "class_name": LABEL_MAP_RAW.get(label, f"unknown_{label}"),
            "PID": pid,
            "image": image,
            "tumorMask": tumor_mask,
            "tumorBorder": tumor_border,
            "file": filepath.name,
        }
    except Exception:
        # Fallback to h5py for v7.3 MAT files
        import h5py
        with h5py.File(filepath, "r") as f:
            cjdata = f["cjdata"]
            label = int(cjdata["label"][0, 0])
            pid_raw = cjdata["PID"][:]
            pid = "".join(chr(c[0]) for c in pid_raw)
            image = np.array(cjdata["image"], dtype=np.float32).T
            tumor_mask = np.array(cjdata["tumorMask"], dtype=np.uint8).T
            return {
                "label": label,
                "class_name": LABEL_MAP_RAW.get(label, f"unknown_{label}"),
                "PID": pid,
                "image": image,
                "tumorMask": tumor_mask,
                "file": filepath.name,
            }

def validate_dataset(raw_dir: Path) -> Dict[str, Any]:
    """
    Validate all .mat files in raw_dir:
    - Counts images per class
    - Validates mask presence and shapes
    - Detects missing/corrupted files
    """
    mat_files = sorted(raw_dir.glob("*.mat"))
    # Filter out cvind.mat
    mat_files = [f for f in mat_files if f.name != "cvind.mat"]
    
    total = len(mat_files)
    print(f"\n[VALIDATE] Found {total} .mat data files in {raw_dir}...")
    
    class_counts = {"meningioma": 0, "glioma": 0, "pituitary": 0, "unknown": 0}
    patients = set()
    corrupted = []
    shapes = set()
    mask_pixels_count = 0
    
    for idx, f in enumerate(mat_files):
        if (idx + 1) % 500 == 0 or idx == total - 1:
            print(f"  Validating file {idx + 1}/{total}...")
        try:
            sample = load_single_mat(f)
            cname = sample["class_name"]
            class_counts[cname] = class_counts.get(cname, 0) + 1
            patients.add(sample["PID"])
            shapes.add(sample["image"].shape)
            mask_pixels = int(sample["tumorMask"].sum())
            if mask_pixels > 0:
                mask_pixels_count += 1
        except Exception as e:
            corrupted.append({"file": f.name, "error": str(e)})

    summary = {
        "total_images": total,
        "unique_patients": len(patients),
        "class_distribution": class_counts,
        "image_shapes": [list(s) for s in shapes],
        "images_with_valid_tumor_masks": mask_pixels_count,
        "corrupted_files_count": len(corrupted),
        "corrupted_files": corrupted[:10],
    }

    print("\n" + "=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Valid Images: {total}")
    print(f"Unique Patients:    {len(patients)}")
    print(f"Image Dimensions:   {[list(s) for s in shapes]}")
    print(f"Masks Available:    {mask_pixels_count}/{total} images")
    print("Class Distribution:")
    for k, v in class_counts.items():
        pct = (v / total * 100) if total > 0 else 0
        print(f"  - {k.capitalize():12s}: {v:5d} ({pct:5.2f}%)")
    print(f"Corrupted Files:    {len(corrupted)}")
    print("=" * 60)

    with open(DATASET_SUMMARY_PATH, "w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"[SUMMARY] Saved dataset summary to {DATASET_SUMMARY_PATH}")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Download and validate Figshare Brain Tumor Dataset")
    parser.add_argument("--parts", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Parts to download (1, 2, 3, 4)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: download part 1 only (766 images)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip download and validate existing files only")
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.validate_only:
        parts_to_get = [1] if args.quick else args.parts
        print(f"[START] Dataset download requested for parts: {parts_to_get}")

        # Download zips
        for item in FIGSHARE_FILES:
            if item["part"] == 0 or item["part"] in parts_to_get:
                dest = RAW_DATA_DIR / item["name"]
                success = download_file(item["url"], dest, item["size"])
                if not success:
                    print(f"[WARNING] Could not download {item['name']}. Continuing...")
                elif item["name"].endswith(".zip"):
                    extract_zip(dest, RAW_DATA_DIR)

    # Validate dataset
    summary = validate_dataset(RAW_DATA_DIR)
    if summary["total_images"] == 0:
        print("[NOTICE] No dataset images currently in data/raw.")
        print("To download manually:")
        print("  Source: Figshare Brain Tumor Dataset (Cheng et al., 2017)")
        print("  URL: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427")
        print("  Files: brainTumorDataPublic_1-766.zip, brainTumorDataPublic_767-1532.zip, etc.")
        print("  Extract all .mat files to: data/raw/")
        sys.exit(1)
    else:
        print("[COMPLETE] Dataset is ready.")

if __name__ == "__main__":
    main()
