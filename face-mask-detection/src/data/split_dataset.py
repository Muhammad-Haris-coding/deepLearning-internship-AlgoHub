"""
Dataset Splitter
----------------
Splits the raw dataset into training, validation,
and testing sets.

Author : Afaq Ahmad Khan
Project: Face Mask Detection
"""

import sys
from pathlib import Path
import random
import shutil

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CLASS_NAMES,
    RANDOM_STATE,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
)

def create_directories():
    """
    Create processed dataset directories.
    """

    for split in ("train", "valid", "test"):
        for class_name in CLASS_NAMES:
            directory = PROCESSED_DATA_DIR / split / class_name
            directory.mkdir(parents=True, exist_ok=True)


def copy_images(images, destination):
    """
    Copy images to destination directory.
    """

    for image_path in images:
        shutil.copy2(image_path, destination)


def split_class(class_name):
    """
    Split one class into train/valid/test.
    """

    source_dir = RAW_DATA_DIR / class_name

    images = sorted(
        [
            image
            for image in source_dir.iterdir()
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    train_images, temp_images = train_test_split(
        images,
    train_size=TRAIN_RATIO,
    random_state=RANDOM_STATE,
    shuffle=True,
    )

    valid_fraction = TEST_RATIO / (VALID_RATIO + TEST_RATIO)

    valid_images, test_images = train_test_split(
    temp_images,
    test_size=valid_fraction,
    random_state=RANDOM_STATE,
    shuffle=True,
    )

    copy_images(
        train_images,
        PROCESSED_DATA_DIR / "train" / class_name,
    )

    copy_images(
        valid_images,
        PROCESSED_DATA_DIR / "valid" / class_name,
    )

    copy_images(
        test_images,
        PROCESSED_DATA_DIR / "test" / class_name,
    )

    print(f"\n{class_name}")
    print(f"Train : {len(train_images)}")
    print(f"Valid : {len(valid_images)}")
    print(f"Test  : {len(test_images)}")


def main():

    random.seed(RANDOM_STATE)

    create_directories()

    print("=" * 50)
    print("Splitting Dataset")
    print("=" * 50)

    for class_name in CLASS_NAMES:
        split_class(class_name)

    print("\nDataset successfully prepared!")
    print(PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()