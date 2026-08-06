Live Face Verification System
Overview
This repository contains a real-time computer vision application designed to perform face detection, alignment, and biometric verification using live video feeds. The system extracts 128-dimensional facial embeddings and calculates the Euclidean distance between vectors to achieve 1:1 identity verification. The frontend is served via a browser-based Streamlit interface.

System Architecture
The application pipeline consists of four primary stages:

Video Ingestion: Hardware-level camera access and real-time frame extraction utilizing OpenCV.

Face Detection & Alignment: Identifying facial bounding boxes and mapping landmarks to standardize the input crop.

Feature Extraction: Passing the aligned face through a pre-trained ResNet deep convolutional neural network (via Dlib) to generate a 128-element continuous vector (facial encoding).

Distance Metric & Verification: Comparing the live vector against a repository of known embeddings using a configurable Euclidean distance threshold.

Technical Stack
Language: Python 3.8+

Core Libraries:

dlib (C++ backend for machine learning models)

face_recognition (Facial embedding extraction)

opencv-python (Frame processing and visual rendering)

numpy (Vector calculations)

Frontend: streamlit

Prerequisites
Because the underlying dlib library requires native C++ compilation, you must have the appropriate build tools installed on your system prior to setting up the Python environment.

For Windows Users:

Install Visual Studio Build Tools (Select the "Desktop development with C++" workload during installation).

Install CMake globally via the command line: pip install cmake.

For Linux/macOS Users:
Ensure you have cmake and a C++ compiler (like gcc or clang) installed via your package manager.

Installation
Clone the repository:

Bash
git clone https://github.com/YourUsername/faceRecognition.git
cd faceRecognition
Create and activate a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Usage
Configure Known Identities:

Create a directory named known_faces in the root of the project.

Add a clear, forward-facing image of the subject to this directory (e.g., reference_subject.jpg).

Update the path and label mapping in the load_known_faces() function within app.py.

Execute the Application:

Bash
streamlit run app.py
Operation:

Access the local server URL provided in the terminal (typically http://localhost:8501).

Toggle the "Start Webcam Stream" parameter to initialize the inference pipeline.

Author
Muhammad Haris
Computer Vision & Deep Learning Engineer | AI Specialist