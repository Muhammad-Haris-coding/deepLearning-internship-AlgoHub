"""
Helper Utilities
----------------
Common utility functions used throughout the project.

Author : Afaq Ahmad Khan
Project : Face Mask Detection
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# ==========================================================
# Image Loading
# ==========================================================

def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    image_path : str | Path

    Returns
    -------
    np.ndarray
        RGB image.

    Raises
    ------
    FileNotFoundError
    """

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==========================================================
# Color Conversion
# ==========================================================

def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR.
    """

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB.
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==========================================================
# Draw Prediction
# ==========================================================

def draw_prediction(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> np.ndarray:
    """
    Draw bounding box and prediction on image.

    Parameters
    ----------
    frame : np.ndarray

    bbox : tuple
        (x, y, w, h)

    label : str

    confidence : float

    Returns
    -------
    np.ndarray
    """

    x, y, w, h = bbox

    if label.lower() == "mask":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2,
    )

    text = f"{label} ({confidence:.2f}%)"

    text_y = max(25, y - 10)

    cv2.putText(
        frame,
        text,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    return frame


# ==========================================================
# FPS Counter
# ==========================================================

def draw_fps(
    frame: np.ndarray,
    fps: float,
) -> np.ndarray:
    """
    Draw FPS on image.
    """

    cv2.putText(
        frame,
        f"FPS : {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame


# ==========================================================
# Resize While Keeping Aspect Ratio
# ==========================================================

def resize_image(
    image: np.ndarray,
    width: int = 800,
) -> np.ndarray:
    """
    Resize image while preserving aspect ratio.
    """

    h, w = image.shape[:2]

    ratio = width / w

    new_height = int(h * ratio)

    return cv2.resize(
        image,
        (width, new_height),
    )