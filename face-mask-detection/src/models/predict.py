"""
Prediction Module
-----------------
Loads the trained model and performs inference on
a single face image.

Author : Afaq Ahmad Khan
Project : Face Mask Detection
"""

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf

from src.config import (
    IMAGE_SIZE,
    MODEL_PATH,
    MASK_LABEL,
    NO_MASK_LABEL,
)

# ==========================================================
# Global Model Cache
# ==========================================================

_MODEL = None


# ==========================================================
# Load Model
# ==========================================================

def load_model() -> tf.keras.Model:
    """
    Load TensorFlow model once.
    """

    global _MODEL

    if _MODEL is None:

        model_path = Path(MODEL_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{model_path}"
            )

        _MODEL = tf.keras.models.load_model(model_path)

    return _MODEL


# ==========================================================
# Preprocess Image
# ==========================================================

def preprocess_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Prepare RGB image for inference.
    """

    if image is None:
        raise ValueError("Image is None.")

    if image.ndim != 3:
        raise ValueError("Expected RGB image.")

    image = cv2.resize(
        image,
        IMAGE_SIZE,
    )

    image = image.astype(np.float32)

    image /= 255.0

    image = np.expand_dims(
        image,
        axis=0,
    )

    return image


# ==========================================================
# Prediction
# ==========================================================

def predict(
    model: tf.keras.Model,
    image: np.ndarray,
) -> Dict[str, float]:
    """
    Predict mask status.
    """

    image = preprocess_image(image)

    probability = float(
        model.predict(
            image,
            verbose=0,
        )[0][0]
    )

    if probability < 0.5:

        label = MASK_LABEL
        confidence = (1.0 - probability) * 100

    else:

        label = NO_MASK_LABEL
        confidence = probability * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "probability": round(probability, 4),
    }


# ==========================================================
# Predict From Path
# ==========================================================

def predict_from_path(
    image_path: str,
) -> Dict[str, float]:

    model = load_model()

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return predict(
        model,
        image,
    )


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    result = predict_from_path("sample.jpg")

    print(result)