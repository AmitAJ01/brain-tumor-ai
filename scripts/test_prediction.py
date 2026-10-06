"""
Inference Verification Script
Loads a real brain MRI slice and performs complete end-to-end inference
via the BrainTumorAnalysisFacade without requiring the web frontend.
"""

import sys
import json
import base64
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.application.brain_tumor_facade import BrainTumorAnalysisFacade
from backend.app.infrastructure.repositories.dataset_repository import DatasetRepository

def run_test_prediction(sample_id: str = None):
    print("=" * 65)
    print("BRAINTUMORAI - STANDALONE INFERENCE VERIFICATION TEST")
    print("=" * 65)

    dataset_repo = DatasetRepository()
    manifest_path = PROJECT_ROOT / "data" / "processed" / "dataset_manifest.json"

    if not manifest_path.exists():
        print("[ERROR] Dataset manifest not found. Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    with open(manifest_path, "r") as mf:
        manifest = json.load(mf)

    # Pick a sample from test set
    test_samples = [s for s in manifest if s["split"] == "test"]
    if not test_samples:
        test_samples = manifest

    if sample_id is None:
        target_sample = test_samples[0]
    else:
        matching = [s for s in manifest if s["id"] == str(sample_id)]
        target_sample = matching[0] if matching else test_samples[0]

    img_path = PROJECT_ROOT / target_sample["image_path"]
    true_label = target_sample["class_name"]
    pid = target_sample.get("patient_id", "Unknown")

    print(f"Test Image:     {img_path.name}")
    print(f"Ground Truth:   {true_label.upper()}")
    print(f"Patient ID:     {pid}")
    print("-" * 65)

    # Initialize Facade
    facade = BrainTumorAnalysisFacade()

    # Test 1: Standard Preprocessing Strategy
    print("\n[TEST 1] Running Analysis with Standard Preprocessing Strategy...")
    raw_img = Image.open(img_path)
    res_std = facade.analyze(raw_img, strategy_name="standard")

    print(f"  Predicted Class:       {res_std['prediction'].upper()}")
    print(f"  Confidence:            {res_std['confidence_percentage']}%")
    print(f"  Processing Time:       {res_std['processing_time_ms']} ms")
    print("  Probabilities:")
    for cls_name, prob in res_std["probabilities"].items():
        bar = "#" * int(prob * 20)
        print(f"    - {cls_name.capitalize():12s}: {prob:6.4f} [{bar:<20s}]")

    # Verify Grad-CAM
    if res_std.get("gradcam_url"):
        print("  Grad-CAM:              Generated successfully (Base64 data URL present)")
        # Save a copy to results for visual verification
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        gcam_b64 = res_std["gradcam_url"].split(",")[1]
        with open(results_dir / "test_gradcam_overlay.png", "wb") as gf:
            gf.write(base64.b64decode(gcam_b64))
        print(f"  Grad-CAM Preview:      Saved to results/test_gradcam_overlay.png")

    # Verify Segmentation
    seg = res_std.get("segmentation", {})
    if seg.get("available"):
        print(f"  Segmentation Mask:     Available")
        print(f"  Tumor Detected:        {seg['tumor_detected']}")
        print(f"  Tumor Pixel Area:      {seg['tumor_area_pixels']} px")
        print(f"  Tumor Area % of Brain: {seg['tumor_area_percentage']}%")
    else:
        print("  Segmentation Mask:     Model weights not yet trained (Modular placeholder ready)")

    # Test 2: Fuzzy Preprocessing Strategy
    print("\n[TEST 2] Switching Strategy to Fuzzy Preprocessing (Hassan & Boulila 2025)...")
    res_fuzzy = facade.analyze(raw_img, strategy_name="fuzzy")
    print(f"  Strategy Active:       {res_fuzzy['strategy_used']}")
    print(f"  Predicted Class:       {res_fuzzy['prediction'].upper()}")
    print(f"  Confidence:            {res_fuzzy['confidence_percentage']}%")
    print(f"  Processing Time:       {res_fuzzy['processing_time_ms']} ms")

    print("\n" + "=" * 65)
    print("INFERENCE TEST PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    run_test_prediction(sid)
