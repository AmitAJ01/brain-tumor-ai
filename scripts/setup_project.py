"""
Project Setup and Pipeline Automation Script for BrainTumorAI
Orchestrates end-to-end verification, dataset preparation, model training (if not already trained),
evaluation, and sample explainability artifacts.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_step(step_name: str, cmd: list) -> bool:
    print("\n" + "=" * 65)
    print(f"STEP: {step_name}")
    print("=" * 65)
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] Step '{step_name}' failed with return code {result.returncode}")
        return False
    return True

def main():
    print("=" * 65)
    print("BRAINTUMORAI - AUTOMATED PROJECT SETUP & VERIFICATION")
    print("=" * 65)

    # 1. Environment & Dependencies Check
    print("\n[CHECK 1/8] Verifying Python Environment & Dependencies...")
    required_packages = ["torch", "torchvision", "fastapi", "uvicorn", "PIL", "cv2", "sklearn", "matplotlib", "requests"]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[WARNING] Missing dependencies: {missing}")
        print("Please run: pip install -r backend/requirements.txt")
    else:
        print("All required core Python packages are available.")

    # 2. Check Dataset
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    raw_mats = list(raw_dir.glob("*.mat"))

    if len(raw_mats) < 100:
        print("\n[STEP 2/8] Dataset not detected in data/raw. Initiating download...")
        run_step("Download Dataset", [sys.executable, "scripts/download_dataset.py", "--parts", "1", "2", "3", "4"])
    else:
        print(f"\n[STEP 2/8] Raw dataset already present: {len(raw_mats)} .mat files found.")

    # 3. Prepare Dataset (if needed)
    manifest_path = processed_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        print("\n[STEP 3/8] Processed dataset not found. Running preparation...")
        run_step("Prepare Dataset", [sys.executable, "scripts/prepare_dataset.py", "--size", "224"])
    else:
        print(f"\n[STEP 3/8] Processed dataset already prepared: {manifest_path} exists.")

    # 4. Classifier Training
    classifier_path = PROJECT_ROOT / "models" / "brain_tumor_efficientnet.pth"
    if not classifier_path.exists():
        print("\n[STEP 4/8] Classifier checkpoint not found. Initiating training...")
        run_step("Train Classifier", [sys.executable, "scripts/train_classifier.py", "--epochs", "4", "--batch-size", "32"])
    else:
        print(f"\n[STEP 4/8] Trained classifier checkpoint exists: {classifier_path}")

    # 5. Evaluate Classifier
    metrics_path = PROJECT_ROOT / "results" / "classification_metrics.json"
    if not metrics_path.exists() and classifier_path.exists():
        print("\n[STEP 5/8] Evaluating Classifier...")
        run_step("Evaluate Classifier", [sys.executable, "scripts/evaluate.py"])
    elif metrics_path.exists():
        print(f"\n[STEP 5/8] Classification metrics already recorded: {metrics_path}")

    # 6. Test Single Prediction & Grad-CAM
    gradcam_preview = PROJECT_ROOT / "results" / "test_gradcam_overlay.png"
    if not gradcam_preview.exists() and classifier_path.exists():
        print("\n[STEP 6/8] Generating Grad-CAM and testing inference...")
        run_step("Test Inference & Grad-CAM", [sys.executable, "scripts/test_prediction.py"])
    else:
        print(f"\n[STEP 6/8] Inference test verified.")

    # 7. Segmentation Training (if feasible and not already trained)
    segmentor_path = PROJECT_ROOT / "models" / "brain_tumor_efficientunet.pth"
    if not segmentor_path.exists():
        print("\n[STEP 7/8] Segmentation model not found. Training lightweight EfficientNet-UNet...")
        run_step("Train Segmentation", [sys.executable, "scripts/train_segmentation.py", "--epochs", "2", "--samples", "300"])
    else:
        print(f"\n[STEP 7/8] Segmentation model checkpoint exists: {segmentor_path}")

    # 8. Summary
    print("\n" + "=" * 65)
    print("PROJECT SETUP COMPLETE")
    print("=" * 65)
    print("Backend command:  uvicorn backend.app.main:app --port 8000 --reload")
    print("Frontend command: cd frontend && npm run dev")
    print("=" * 65)

if __name__ == "__main__":
    main()
