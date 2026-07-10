import streamlit as st

from src.pages.about import about_page
from src.pages.home import home_page
from src.pages.prediction import prediction_page

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="",
    layout="wide",
)

st.sidebar.title("Navigation")

pages = {
    "Home": home_page,
    "Prediction": prediction_page,
    "About": about_page,
}

page = st.sidebar.radio("Go To", list(pages.keys()))
pages[page]()