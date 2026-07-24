import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import inference
from inference import DEFAULT_MODEL_PATH, _load_alexnet, predict_image
from utils import get_device

st.set_page_config(page_title="Plant Disease Prediction", page_icon="🌿", layout="centered")

st.title("Plant Disease Prediction using AlexNet")
st.write("AI-based plant leaf disease classification using deep learning.")


@st.cache_resource
def load_model():
    return _load_alexnet(DEFAULT_MODEL_PATH, get_device())


def predict_uploaded_image(image: Image.Image) -> dict:
    cached = load_model()
    original_loader = inference._load_alexnet
    inference._load_alexnet = lambda model_path, device: cached
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.convert("RGB").save(tmp.name)
            return predict_image(tmp.name)
    finally:
        inference._load_alexnet = original_loader


uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            result = predict_uploaded_image(image)

        st.subheader("Prediction")
        st.write(result["class_name"])

        st.subheader("Disease name")
        st.write(result["class_name"])

        st.subheader("Confidence")
        st.write(f"{result['confidence'] * 100:.2f}%")

        st.subheader("Top Predictions")
        table = pd.DataFrame(
            [
                {
                    "Rank": rank,
                    "Disease": pred["class_name"],
                    "Probability": f"{pred['probability'] * 100:.2f}%",
                }
                for rank, pred in enumerate(result["top_predictions"], start=1)
            ]
        )
        st.table(table)
