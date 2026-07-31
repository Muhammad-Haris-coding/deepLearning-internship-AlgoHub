# Crop Disease Prediction

## Project Overview

This project provides a Streamlit-based application for performing inference with a pre-trained ResNet50 model on the PlantDoc dataset. The model predicts plant diseases from leaf images across 27 classes.

## Features

- Load a trained ResNet50 model (state_dict) and run inference.
- Preprocessing consistent with ResNet50 ImageNet normalization.
- Streamlit app with a polished UI, top-5 predictions, and probability table.

## Folder Structure

CropDiseasePrediction/

- app.py
- inference.py
- best_plantdoc_model.pth
- requirements.txt
- README.md
- utils/
  - class_names.py
- assets/ (optional)
- sample_images/ (optional)

## Installation

1. Create a Python 3.11 virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Ensure `best_plantdoc_model.pth` is in the project root. Then run:

```bash
streamlit run app.py
```

The app will open in a browser where you can upload leaf images and get predictions.

## Screenshots

Placeholders for screenshots of the Streamlit app.

## Future Improvements

- Add batch inference for multiple images.
- Add model explainability (Grad-CAM) to show attention maps.
- Add Dockerfile for containerized deployment.
