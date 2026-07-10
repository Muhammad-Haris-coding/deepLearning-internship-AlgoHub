"""
preprocessing.py
----------------
This module contains functions for preprocessing the MNIST dataset
and uploaded images for inference.
"""

import numpy as np


def preprocess_mnist_data(X_train, X_test):
    """
    Preprocess the MNIST dataset.

    Steps:
    1. Reshape images to (28, 28, 1)
    2. Convert datatype to float32
    3. Normalize pixel values to [0, 1]

    Returns:
        X_train, X_test
    """

    # Reshape images to match the CNN input format.
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # Convert data to float32.
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    # Normalize pixel values to the [0, 1] range.
    X_train /= 255.0
    X_test /= 255.0

    return X_train, X_test


def display_dataset_info(X_train, X_test):
    """Display dataset information after preprocessing."""

    print("=" * 50)
    print("Dataset After Preprocessing")
    print("=" * 50)

    print(f"Training Shape : {X_train.shape}")
    print(f"Testing Shape  : {X_test.shape}")

    print(f"\nTraining Data Type : {X_train.dtype}")
    print(f"Testing Data Type  : {X_test.dtype}")

    print(f"\nTraining Min Pixel : {X_train.min()}")
    print(f"Training Max Pixel : {X_train.max()}")

    print(f"\nTesting Min Pixel : {X_test.min()}")
    print(f"Testing Max Pixel : {X_test.max()}")