import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dataset import get_val_loader
from model import load_model_from_checkpoint
from utils import get_device, load_classes


def collect_predictions(model: nn.Module, loader, device: torch.device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def evaluate(model_path: str | Path, batch_size: int = 32) -> None:
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = None
    if isinstance(checkpoint, dict) and "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
    if class_names is None:
        class_names = load_classes()

    model = load_model_from_checkpoint(model_path, num_classes=len(class_names), device=device)
    val_loader = get_val_loader(batch_size=batch_size)
    labels, preds = collect_predictions(model, val_loader, device)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(labels, preds)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained plant disease model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g. results/best_model.pth)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model_path, batch_size=args.batch_size)
