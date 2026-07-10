import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.predict import load_trained_model, predict_digit, preprocess_uploaded_image


@st.cache_resource
def load_model():
    return load_trained_model()


model = load_model()


def show_prediction(image, caption):
    # Convert RGBA to RGB if needed.
    if image.mode == "RGBA":
        image = image.convert("RGB")

    st.image(image, caption=caption, width=250)

    processed_image = preprocess_uploaded_image(image)

    digit, confidence, probabilities = predict_digit(model, processed_image)

    st.success(f"Predicted Digit: {digit}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ["steelblue"] * 10
    colors[digit] = "green"

    ax.bar(range(10), probabilities, color=colors)
    ax.set_xticks(range(10))
    ax.set_xlabel("Digits")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)

    st.pyplot(fig)


def prediction_page():
    st.title("Handwritten Digit Recognition")

    option = st.radio(
        "Choose Prediction Method",
        ["Upload Image", "Draw Digit"],
        horizontal=True,
    )

    # Upload Image
    if option == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a handwritten digit",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            show_prediction(image, "Uploaded Image")

    # Draw Digit
    else:
        st.subheader("Draw a Digit")
        st.write("Draw a black digit on the white canvas and click Predict.")

        brush_size = st.slider("Brush Size", 5, 25, 10)

        canvas_result = st_canvas(
            stroke_width=brush_size,
            stroke_color="black",
            background_color="white",
            width=380,
            height=380,
            drawing_mode="freedraw",
            fill_color="rgba(255,255,255,0)",
            update_streamlit=True,
            key="digit_canvas",
        )

        if st.button("Predict Drawn Digit"):
            if canvas_result is None or canvas_result.image_data is None:
                st.warning("Please draw a digit first.")
                return

            image_array = canvas_result.image_data.astype(np.uint8)

            # Check whether the canvas is effectively blank.
            if np.all(image_array[:, :, :3] == 255):
                st.warning("Please draw a digit first.")
                return

            image = Image.fromarray(image_array).convert("RGB")
            show_prediction(image, "Drawn Digit")