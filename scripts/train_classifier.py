"""
EfficientNet-B0 Classifier Training Script for BrainTumorAI
Adheres to methodology from Tiwary et al. (2025) and Sultan et al. (2019).

Features:
- Transfer learning with ImageNet-pretrained EfficientNet-B0
- Stratified train/val/test evaluation
- Data augmentations: Horizontal/Vertical flips, Rotation, Mild Jitter
- Real metric calculation: Accuracy, Precision, Recall, F1, Confusion Matrix
- Plots: training_history.png and confusion_matrix.png
- Zero fabricated metrics
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class BrainMRIDataset(Dataset):
    """PyTorch Dataset loading processed PNG slices and labels."""
    def __init__(self, samples: List[Dict[str, Any]], root_dir: Path, transform=None):
        self.samples = samples
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = self.root_dir / item["image_path"]
        img = Image.open(img_path).convert("RGB")
        label = int(item["label"])

        if self.transform:
            img = self.transform(img)

        return img, label, item["id"]

def get_transforms(image_size: int = 224):
    """Augmentations as highlighted in Tiwary et al. (2025) and Sultan et al. (2019)."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: Path):
    """Plot and save confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Brain Tumor Classification - Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_history(history: Dict[str, List[float]], save_path: Path):
    """Plot and save training and validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Loss
    ax1.plot(history["train_loss"], label="Train Loss", color="#2563eb", lw=2)
    ax1.plot(history["val_loss"], label="Val Loss", color="#dc2626", lw=2)
    ax1.set_title("Loss History")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Accuracy
    ax2.plot(history["train_acc"], label="Train Acc", color="#2563eb", lw=2)
    ax2.plot(history["val_acc"], label="Val Acc", color="#16a34a", lw=2)
    ax2.set_title("Accuracy History")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def train_classifier(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cpu",
    max_train_samples: Optional[int] = None
):
    set_seed(42)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "classifier").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = PROCESSED_DIR / "dataset_manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] Manifest {manifest_path} not found. Run prepare_dataset.py first.")
        sys.exit(1)

    with open(manifest_path, "r") as mf:
        all_samples = json.load(mf)

    train_samples = [s for s in all_samples if s["split"] == "train"]
    val_samples = [s for s in all_samples if s["split"] == "val"]
    test_samples = [s for s in all_samples if s["split"] == "test"]

    if max_train_samples and max_train_samples < len(train_samples):
        print(f"[TRAIN] Subsampling train set to {max_train_samples} samples for rapid training.")
        random.shuffle(train_samples)
        train_samples = train_samples[:max_train_samples]

    print("\n" + "=" * 60)
    print("TRAINING SETUP")
    print("=" * 60)
    print(f"Architecture:      EfficientNet-B0 (Transfer Learning)")
    print(f"Classes (3):       {CLASS_NAMES}")
    print(f"Device:            {device}")
    print(f"Epochs:            {epochs}")
    print(f"Batch Size:        {batch_size}")
    print(f"Train Samples:     {len(train_samples)}")
    print(f"Val Samples:       {len(val_samples)}")
    print(f"Test Samples:      {len(test_samples)}")
    print("=" * 60)

    train_trans, eval_trans = get_transforms(224)

    train_ds = BrainMRIDataset(train_samples, PROJECT_ROOT, transform=train_trans)
    val_ds = BrainMRIDataset(val_samples, PROJECT_ROOT, transform=eval_trans)
    test_ds = BrainMRIDataset(test_samples, PROJECT_ROOT, transform=eval_trans)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Instantiate model
    print("[MODEL] Loading ImageNet pretrained EfficientNet-B0...")
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(CLASS_NAMES))
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "brain_tumor_efficientnet.pth"
    best_classifier_sub_path = MODELS_DIR / "classifier" / "brain_tumor_efficientnet.pth"

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        # Validation pass
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = (val_correct / max(val_total, 1)) * 100.0

        scheduler.step(val_loss)

        history["train_loss"].append(round(epoch_loss, 4))
        history["train_acc"].append(round(epoch_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))

        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - {elapsed:.1f}s | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:5.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": CLASS_NAMES
            }
            torch.save(checkpoint_payload, best_model_path)
            torch.save(checkpoint_payload, best_classifier_sub_path)
            print(f"  --> Saved new best checkpoint to {best_model_path.name} (Val Acc: {val_acc:.2f}%)")

    total_training_sec = time.time() - start_training_time
    print(f"\n[TRAINING FINISHED] Total time: {total_training_sec:.1f}s. Best Val Acc: {best_val_acc:.2f}%")

    # Evaluate on unseen Test Set
    print("\n[TEST EVALUATION] Evaluating best checkpoint on unseen Test Set...")
    best_ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate actual metrics
    acc = accuracy_score(all_targets, all_preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    report_text = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)

    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    per_class_metrics = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class_metrics[name] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_supp[i])
        }

    test_metrics = {
        "model": "EfficientNet-B0",
        "num_classes": 3,
        "classes": CLASS_NAMES,
        "test_samples": len(all_targets),
        "test_accuracy": round(float(acc), 4),
        "test_accuracy_percentage": round(float(acc) * 100.0, 2),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "training_epochs": epochs,
        "total_training_time_seconds": round(total_training_sec, 2),
        "history": history
    }

    # Save results files
    metrics_path = RESULTS_DIR / "classification_metrics.json"
    with open(metrics_path, "w") as jf:
        json.dump(test_metrics, jf, indent=2)

    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as rf:
        rf.write("BrainTumorAI - Classification Evaluation Report\n")
        rf.write(f"Model: EfficientNet-B0 | Dataset: Figshare Brain Tumor (Cheng et al.)\n")
        rf.write(f"Test Accuracy: {acc * 100.0:.2f}%\n")
        rf.write(f"Macro F1-Score: {f1_macro:.4f}\n\n")
        rf.write(report_text)
        rf.write("\nConfusion Matrix:\n")
        rf.write(np.array2string(cm))

    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)

    history_path = RESULTS_DIR / "training_history.png"
    plot_history(history, history_path)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION METRICS (ACTUAL EXECUTION)")
    print("=" * 60)
    print(f"Test Accuracy:       {acc * 100.0:.2f}%")
    print(f"Macro Precision:     {prec_macro:.4f}")
    print(f"Macro Recall:        {rec_macro:.4f}")
    print(f"Macro F1-Score:      {f1_macro:.4f}")
    print("\nClassification Report:\n" + report_text)
    print(f"Metrics saved:       {metrics_path}")
    print(f"Report saved:        {report_path}")
    print(f"Confusion Matrix:    {cm_path}")
    print(f"History Plot:        {history_path}")
    print("=" * 60)

    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max train samples")
    args = parser.parse_args()

    train_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        max_train_samples=args.max_samples
    )
