"""
Visualization Utilities
-----------------------
Utility functions for plotting training history and
confusion matrix.

Author : Afaq Ahmad Khan
Project : Face Mask Detection
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config import OUTPUT_DIR


def plot_training_history(history) -> None:
    """
    Plot training and validation accuracy/loss.

    Parameters
    ----------
    history : keras.callbacks.History
        Model training history.
    """

    history = history.history

    # Accuracy Plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["accuracy"],
        label="Training Accuracy",
        linewidth=2,
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2,
    )

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    accuracy_path = OUTPUT_DIR / "accuracy_curve.png"

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Loss Plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["loss"],
        label="Training Loss",
        linewidth=2,
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss",
        linewidth=2,
    )

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    loss_path = OUTPUT_DIR / "loss_curve.png"

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Accuracy plot saved to: {accuracy_path}")
    print(f"Loss plot saved to: {loss_path}")


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
) -> None:
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray

    y_pred : np.ndarray

    class_names : list
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )

    plt.title("Confusion Matrix")

    save_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Confusion matrix saved to: {save_path}")