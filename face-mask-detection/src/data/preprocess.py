"""
Data Preprocessing
------------------
Creates TensorFlow datasets for training, validation,
and testing.

Author : Afaq Ahmad Khan
Project: Face Mask Detection
"""

from typing import Tuple
import sys
import tensorflow as tf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_STATE,
)

AUTOTUNE = tf.data.AUTOTUNE


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset(
    directory,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Load image dataset from directory.

    Parameters
    ----------
    directory : Path
        Dataset directory.

    shuffle : bool
        Shuffle dataset.

    Returns
    -------
    tf.data.Dataset
    """

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=RANDOM_STATE,
    )

    return dataset


# ==========================================================
# Normalize Images
# ==========================================================

def normalize(
    images: tf.Tensor,
    labels: tf.Tensor,
):
    """
    Normalize pixel values to [0,1].
    """

    images = tf.cast(images, tf.float32) / 255.0

    return images, labels


# ==========================================================
# Prepare Dataset
# ==========================================================

def prepare_dataset(
    dataset: tf.data.Dataset,
    training: bool = False,
) -> tf.data.Dataset:
    """
    Prepare dataset pipeline.

    Operations:
        • Normalize
        • Cache
        • Shuffle (training only)
        • Prefetch
    """

    dataset = dataset.map(
        normalize,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.cache()

    if training:
        dataset = dataset.shuffle(1000)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# ==========================================================
# Load All Datasets
# ==========================================================

def get_datasets() -> Tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    list,
]:
    """
    Returns
    -------
    train_ds
    valid_ds
    test_ds
    class_names
    """

    train_ds = load_dataset(
        TRAIN_DIR,
        shuffle=True,
    )

    valid_ds = load_dataset(
        VALID_DIR,
        shuffle=False,
    )

    test_ds = load_dataset(
        TEST_DIR,
        shuffle=False,
    )

    class_names = train_ds.class_names

    train_ds = prepare_dataset(
        train_ds,
        training=True,
    )

    valid_ds = prepare_dataset(
        valid_ds,
        training=False,
    )

    test_ds = prepare_dataset(
        test_ds,
        training=False,
    )

    return (
        train_ds,
        valid_ds,
        test_ds,
        class_names,
    )


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    train_ds, valid_ds, test_ds, class_names = get_datasets()

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)

    print(f"Classes           : {class_names}")

    print(
        f"Train Batches     : "
        f"{tf.data.experimental.cardinality(train_ds).numpy()}"
    )

    print(
        f"Validation Batches: "
        f"{tf.data.experimental.cardinality(valid_ds).numpy()}"
    )

    print(
        f"Test Batches      : "
        f"{tf.data.experimental.cardinality(test_ds).numpy()}"
    )