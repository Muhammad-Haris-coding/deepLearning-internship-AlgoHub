"""
About
-----
Information about the Face Mask Detection project.

Author : Afaq Ahmad Khan
Project : Face Mask Detection
"""

import streamlit as st


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="About",
    layout="wide",
)


# ==========================================================
# Header
# ==========================================================

st.title("About the Project")

st.write(
    """
Face Mask Detection is a Computer Vision application that
uses Deep Learning to identify whether a detected person
is wearing a face mask.

The system performs face detection followed by mask
classification in both uploaded images and live webcam
streams.
"""
)

st.divider()


# ==========================================================
# Project Objective
# ==========================================================

st.header("Objective")

st.write(
    """
Develop a real-time face mask detection system capable of
identifying whether a person is wearing a face mask.

The project demonstrates the integration of Computer Vision,
Deep Learning, and real-time video processing using
TensorFlow, MediaPipe, OpenCV, and Streamlit.
"""
)

st.divider()


# ==========================================================
# Workflow
# ==========================================================

st.header("System Workflow")

st.code(
    """
Input Image / Webcam
        │
        ▼
MediaPipe Face Detection
        │
        ▼
Face Cropping
        │
        ▼
Image Preprocessing
        │
        ▼
MobileNetV2 Classification
        │
        ▼
Mask / No Mask Prediction
""",
    language="text",
)

st.divider()


# ==========================================================
# Technology Stack
# ==========================================================

st.header("Technology Stack")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Frameworks")

    st.markdown(
        """
- TensorFlow / Keras
- Streamlit
- MediaPipe
- OpenCV
"""
    )

with col2:

    st.subheader("Libraries")

    st.markdown(
        """
- NumPy
- Pillow
- Matplotlib
- scikit-learn
"""
    )

st.divider()


# ==========================================================
# Dataset
# ==========================================================

st.header("Dataset")

st.write(
    """
The model is trained on a binary image dataset consisting
of two classes:

- With Mask
- Without Mask

The dataset is preprocessed, split into training,
validation, and testing subsets before model training.
"""
)

st.divider()


# ==========================================================
# Model Information
# ==========================================================

st.header("Model")

st.write(
    """
Architecture : MobileNetV2 (Transfer Learning)

Input Size : 224 × 224 × 3

Output : Binary Classification

Activation : Sigmoid

Loss Function : Binary Crossentropy

Optimizer : Adam
"""
)

st.divider()


# ==========================================================
# Features
# ==========================================================

st.header("Features")

st.markdown(
    """
- Image Prediction

- Real-Time Webcam Detection

- Multiple Face Detection

- Confidence Score Display

- Lightweight MobileNetV2 Model

- Streamlit Web Interface
"""
)

st.divider()


# ==========================================================
# Author
# ==========================================================

st.header("Developer")

st.write(
    """
**Name:** Afaq Ahmad Khan

Bachelor of Computer Science

Specialization: Artificial Intelligence

Focus Areas:

- Computer Vision
- Deep Learning
- Machine Learning
- AI Application Development
"""
)

st.divider()


# ==========================================================
# Footer
# ==========================================================

st.caption(
    "Face Mask Detection | Deep Learning Project | Streamlit"
)