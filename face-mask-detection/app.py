"""
Home Page
---------
Landing page for the Face Mask Detection application.

Author : Afaq Ahmad Khan
Project : Face Mask Detection
"""

import streamlit as st


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Face Mask Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# Custom Styling
# ==========================================================

st.markdown(
    """
    <style>

    .main-title{
        font-size:42px;
        font-weight:700;
        color:#1E3A8A;
        margin-bottom:0;
    }

    .subtitle{
        font-size:18px;
        color:#555555;
        margin-top:0;
        margin-bottom:25px;
    }

    .card{
        background-color:#F8F9FA;
        padding:20px;
        border-radius:12px;
        border:1px solid #E5E7EB;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# Header
# ==========================================================

st.markdown(
    '<p class="main-title">Face Mask Detection</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">'
    'Real-Time Face Mask Detection using Deep Learning'
    '</p>',
    unsafe_allow_html=True,
)

st.divider()

# ==========================================================
# Introduction
# ==========================================================

st.write(
    """
This application detects whether a person is wearing a face mask
using a deep learning model based on **MobileNetV2**.

The system combines **MediaPipe Face Detection** with a
TensorFlow/Keras classifier to provide fast and reliable predictions
for uploaded images and live webcam streams.
"""
)

st.divider()

# ==========================================================
# Features
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown("### Features")

    st.markdown(
        """
- Image Prediction
- Real-Time Webcam Detection
- Multiple Face Detection
- Confidence Score
- Fast Inference
"""
    )

with col2:

    st.markdown("### Technology Stack")

    st.markdown(
        """
- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- MediaPipe
- Streamlit
"""
    )

st.divider()

# ==========================================================
# Workflow
# ==========================================================

st.markdown("### Processing Pipeline")

st.code(
    """
Input Image / Webcam
        │
        ▼
MediaPipe Face Detection
        │
        ▼
Crop Face Region
        │
        ▼
MobileNetV2 Classifier
        │
        ▼
Mask / No Mask Prediction
""",
    language="text",
)

st.divider()

# ==========================================================
# Instructions
# ==========================================================

st.markdown("### Getting Started")

st.info(
    """
Use the navigation menu in the left sidebar to access:

• Image Prediction

• Webcam Detection

• About
"""
)

st.divider()

# ==========================================================
# Footer
# ==========================================================

st.caption(
    "Developed using TensorFlow, OpenCV, MediaPipe, and Streamlit."
)