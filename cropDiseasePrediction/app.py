import io
from typing import Tuple, Dict, List

import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from inference import load_model, predict


st.set_page_config(page_title="🌿 Crop Disease Prediction", layout="wide")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #f7f9fb; }
        .prediction-card { border-radius: 10px; padding: 16px; background: linear-gradient(135deg,#e8fff1,#f1f8ff); }
        .small-muted { color: #6b7280; font-size:12px }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model() -> Tuple:
    """Load and cache the model once for the Streamlit session.

    Returns:
        Tuple[torch.nn.Module, torch.device]: model and device
    """
    return load_model()


def sidebar_info() -> None:
    st.sidebar.title("Project Information")
    st.sidebar.markdown("**Model:** ResNet50 (transfer learning)")
    st.sidebar.markdown("**Dataset:** PlantDoc")
    st.sidebar.markdown("**Classes:** 27")
    st.sidebar.markdown("**Framework:** PyTorch")
    st.sidebar.divider()
    st.sidebar.title("Developer")
    st.sidebar.markdown("Built with ❤️ — Crop Disease Prediction demo")


def show_prediction_results(predicted: str, confidence: float, top5: List[tuple], probabilities: Dict[str, float]) -> None:
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("### Prediction")
        st.success(f"{predicted}")
        st.metric(label="Confidence", value=f"{confidence*100:.2f} %")

    with col2:
        st.markdown("### Top 5 Predictions")
        df_top5 = pd.DataFrame(top5, columns=["disease", "probability"])
        df_top5["probability"] = df_top5["probability"] * 100

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(df_top5["disease"][::-1], df_top5["probability"][::-1], color="#2b8cbe")
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()
    st.markdown("### Probabilities")
    df_probs = pd.DataFrame(list(probabilities.items()), columns=["disease", "probability"])  # type: ignore
    df_probs["probability"] = df_probs["probability"] * 100
    df_probs = df_probs.sort_values("probability", ascending=False).reset_index(drop=True)
    st.table(df_probs)


def main() -> None:
    _inject_css()
    sidebar_info()

    st.title("🌿 Crop Disease Prediction")
    st.markdown("Upload a leaf image and the model will predict the most likely disease.")

    model_load_error = None
    try:
        model, device = get_model()
    except Exception as e:
        model = None
        device = None
        model_load_error = e

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"]) 

    if model_load_error:
        st.error(f"Model load failed: {model_load_error}")
        st.stop()

    if uploaded_file is None:
        st.info("Please upload an image to run prediction.")
        return

    try:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    except Exception:
        st.error("Uploaded file is not a valid image or is corrupted.")
        return

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        if model is None:
            st.error("Model not available. Check logs.")
            return

        with st.spinner("Running prediction..."):
            try:
                predicted_class, confidence, top5, probabilities = predict(image)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        show_prediction_results(predicted_class, confidence, top5, probabilities)


if __name__ == "__main__":
    main()
