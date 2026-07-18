"""
Project Configuration
---------------------
This module contains all project-wide constants and directory paths.

Author : Afaq Ahmad Khan
Project: Face Mask Detection
"""

from pathlib import Path

# ==========================================================
# Base Directories
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"

# ==========================================================
# Dataset
# ==========================================================

TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VALID_DIR = PROCESSED_DATA_DIR / "valid"
TEST_DIR = PROCESSED_DATA_DIR / "test"

CLASS_NAMES = (
    "with_mask",
    "without_mask",
)

NUM_CLASSES = len(CLASS_NAMES)

# ==========================================================
# Image Parameters
# ==========================================================

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3

IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# ==========================================================
# Training Parameters
# ==========================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-4

RANDOM_STATE = 42

VALIDATION_SPLIT = 0.2

# ==========================================================
# Dataset Split Ratios
# ==========================================================

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================================
# Model Saving
# ==========================================================

MODEL_NAME = "best_model.keras"

MODEL_PATH = MODEL_DIR / MODEL_NAME

# ==========================================================
# Prediction
# ==========================================================

MASK_LABEL = "Mask"

NO_MASK_LABEL = "No Mask"

CONFIDENCE_THRESHOLD = 0.50

# ==========================================================
# Face Detection
# ==========================================================

FACE_PADDING = 20

# ==========================================================
# Create Directories Automatically
# ==========================================================

for directory in (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    ASSETS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)