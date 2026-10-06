"""
Independent Evaluation Script for BrainTumorAI Models
Evaluates trained checkpoints on the isolated test partition.
Adheres to strict requirement: NEVER FABRICATE ANY METRIC.
All statistics derive directly from model forward passes on real images.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.infrastructure.factories.model_factory import ModelFactory
from backend.app.infrastructure.models.efficientnet_model import CLASS_NAMES
from scripts.train_classifier import BrainMRIDataset, get_transforms, plot_confusion_matrix

def evaluate_models(device: str = "cpu"):
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = PROJECT_ROOT / "data" / "processed" / "dataset_manifest.json"

    if not manifest_path.exists():
        print("[ERROR] Dataset manifest not found. Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    test_samples = [s for s in manifest if s["split"] == "test"]
    print(f"[EVALUATION] Found {len(test_samples)} test samples.")

    # 1. Evaluate Classifier
    weights_path = PROJECT_ROOT / "models" / "brain_tumor_efficientnet.pth"
    if not weights_path.exists():
        weights_path = PROJECT_ROOT / "models" / "classifier" / "brain_tumor_efficientnet.pth"

    if not weights_path.exists():
        print(f"[ERROR] Trained classifier weights not found at {weights_path}. Train model first.")
        sys.exit(1)

    print(f"[EVALUATION] Loading classifier from {weights_path}...")
    classifier = ModelFactory.create("efficientnet", weights_path=str(weights_path), device=device)

    _, eval_trans = get_transforms(224)
    test_ds = BrainMRIDataset(test_samples, PROJECT_ROOT, transform=eval_trans)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []

    classifier.model.eval()
    with torch.no_grad():
        for imgs, targets, _ in test_loader:
            imgs = imgs.to(device)
            outputs = classifier.model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    report_text = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_supp[i])
        }

    eval_data = {
        "model": "EfficientNet-B0",
        "weights_evaluated": str(weights_path),
        "test_samples": len(all_targets),
        "test_accuracy": round(float(acc), 4),
        "test_accuracy_percentage": round(float(acc) * 100.0, 2),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }

    metrics_file = results_dir / "classification_metrics.json"
    with open(metrics_file, "w") as mf:
        json.dump(eval_data, mf, indent=2)

    report_file = results_dir / "classification_report.txt"
    with open(report_file, "w") as rf:
        rf.write("BrainTumorAI - Full Evaluation Report\n")
        rf.write(f"Weights: {weights_path.name}\n")
        rf.write(f"Accuracy: {acc * 100.0:.2f}%\n")
        rf.write(report_text)

    cm_file = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_file)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:           {acc * 100.0:.2f}%")
    print(f"Macro F1-Score:     {f1_macro:.4f}")
    print(f"Weighted F1-Score:  {f1_weighted:.4f}")
    print(f"Classification Report:\n{report_text}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    evaluate_models(args.device)
