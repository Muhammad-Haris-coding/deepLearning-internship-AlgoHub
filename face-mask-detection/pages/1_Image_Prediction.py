"""
Image Prediction Page
---------------------
Upload an image and detect whether detected faces
are wearing a mask.

Author : Afaq Ahmad Khan
Project : Face Mask Detection
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.models.face_detector import FaceDetector
from src.models.predict import (
    load_model,
    predict,
)
from src.utils.helpers import draw_prediction


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Image Prediction",
    layout="wide",
)


# ==========================================================
# Load Resources
# ==========================================================

@st.cache_resource
def get_model():
    """
    Load trained model once.
    """
    return load_model()


@st.cache_resource
def get_face_detector():
    """
    Load MediaPipe Face Detector once.
    """
    return FaceDetector()


MODEL = get_model()
DETECTOR = get_face_detector()


# ==========================================================
# Header
# ==========================================================

st.title("Image Prediction")

st.write(
    """
Upload an image containing one or more faces.
The system will detect each face and classify it
as **Mask** or **No Mask**.
"""
)

st.divider()


# ==========================================================
# File Upload
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)


# ==========================================================
# Prediction
# ==========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    frame = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    faces = DETECTOR.detect(frame)

    result_frame = frame.copy()

    if len(faces) == 0:

        st.warning("No face detected.")

    else:

        for face in faces:

            result = predict(
                MODEL,
                face.image,
            )

            result_frame = draw_prediction(
                frame=result_frame,
                bbox=face.bbox,
                label=result["label"],
                confidence=result["confidence"],
            )

        result_frame = cv2.cvtColor(
            result_frame,
            cv2.COLOR_BGR2RGB,
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True,
            )

        with col2:

            st.subheader("Prediction")

            st.image(
                result_frame,
                use_container_width=True,
            )

        st.success(
            f"Detected {len(faces)} face(s)."
        )


st.divider()

st.caption(
    "Supported formats: JPG, JPEG, PNG"
)