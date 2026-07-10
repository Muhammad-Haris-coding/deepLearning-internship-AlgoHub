import streamlit as st


def about_page():
    st.title("About")

    st.write(
        """
        ## Handwritten Digit Recognition

        This application predicts handwritten digits using
        a Convolutional Neural Network trained on the MNIST dataset.

        ### Features

        - Upload handwritten digit images
        - Real-time prediction
        - Confidence score
        - Probability visualization

        ### Developed Using

        - TensorFlow/Keras
        - Streamlit
        - Python
        """
    )