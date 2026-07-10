import streamlit as st


def home_page():
    st.title("Handwritten Digit Recognition")

    st.markdown("---")

    st.header("Project Overview")

    st.write(
        """
        This project uses a Convolutional Neural Network (CNN)
        trained on the MNIST dataset to recognize handwritten digits.
        """
    )

    st.header("Dataset")

    st.write(
        """
        • 60,000 Training Images

        • 10,000 Testing Images

        • Image Size: 28 × 28 pixels

        • Classes: 0 – 9
        """
    )

    st.header("Technologies Used")

    st.write(
        """
        - TensorFlow/Keras
        - Streamlit
        - NumPy
        - Pillow
        - Matplotlib
        """
    )