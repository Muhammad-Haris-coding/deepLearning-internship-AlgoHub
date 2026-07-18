# Face Mask Detection using Deep Learning

## Overview

This project is a real-time Face Mask Detection system developed using Computer Vision and Deep Learning. It detects human faces from images or a live webcam feed and classifies whether each detected person is wearing a face mask.

The application is built with TensorFlow, MediaPipe, OpenCV, and Streamlit and supports both image-based prediction and real-time webcam detection.

## Project Objectives

* Build a real-time face mask detection system.
* Train a Deep Learning model for binary image classification.
* Detect faces using MediaPipe Face Detection.
* Perform real-time inference on webcam frames.
* Deploy the application using Streamlit.

## Features

* Image Prediction
* Real-Time Webcam Detection
* Multiple Face Detection
* Confidence Score Display
* MobileNetV2 Transfer Learning
* Streamlit Web Interface

## Project Structure

```text
Face-Mask-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
├── outputs/
├── tests/
│
├── src/
│   ├── config.py
│   ├── data/
│   ├── models/
│   └── utils/
│
└── streamlit_app/
    └── pages/
```

## Dataset Structure

```text
data/raw/
├── with_mask/
└── without_mask/
```

## Technologies

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe
* Streamlit
* Streamlit WebRTC
* NumPy
* Pillow
* Matplotlib
* Scikit-learn

## Model

* Architecture: MobileNetV2
* Input Size: 224 × 224 × 3
* Output Layer: Sigmoid
* Loss Function: Binary Crossentropy
* Optimizer: Adam

## Installation

Clone the repository.

```bash
git clone <repository-url>
cd Face-Mask-Detection
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

## Dataset

Place the dataset in:

```text
data/raw/
├── with_mask/
└── without_mask/
```

## Train the Model

```bash
python -m src.models.train
```

After training, the trained model will be saved to the configured model directory.

## Run the Application

```bash
streamlit run app.py
```

The application provides:

* Home Page
* Image Prediction
* Webcam Detection
* About Page

## Workflow

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Training
   │
   ▼
Saved Model
   │
   ▼
Face Detection
   │
   ▼
Prediction
   │
   ▼
Streamlit Application
```
## Author

**Afaq Ahmad Khan**

Bachelor of Computer Science

Specialization: Artificial Intelligence

Focus Areas:

* Computer Vision
* Deep Learning
* Machine Learning
* AI Applications
