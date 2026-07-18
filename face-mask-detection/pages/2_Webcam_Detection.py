"""
Webcam Detection
----------------
Real-time Face Mask Detection using Streamlit WebRTC.

Author : Afaq Ahmad Khan
Project : Face Mask Detection
"""

import logging

import av
import streamlit as st
from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from src.models.face_detector import FaceDetector
from src.models.predict import (
    load_model,
    predict,
)
from src.utils.helpers import draw_prediction

logger = logging.getLogger(__name__)


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Webcam Detection",
    layout="wide",
)


# ==========================================================
# Header
# ==========================================================

st.title("Real-Time Face Mask Detection")

st.write(
    """
Start your webcam to detect whether each detected
person is wearing a face mask.
"""
)

st.divider()


# ==========================================================
# Cached Resources
# ==========================================================

@st.cache_resource
def get_model():
    """
    Load TensorFlow model only once.
    """
    return load_model()


@st.cache_resource
def get_face_detector():
    """
    Load MediaPipe detector only once.
    """
    return FaceDetector()


MODEL = get_model()
FACE_DETECTOR = get_face_detector()


# ==========================================================
# Video Processor
# ==========================================================

class VideoProcessor(VideoProcessorBase):
    """
    Process each webcam frame.
    """

    def __init__(self):
        self.model = MODEL
        self.detector = FACE_DETECTOR

    def recv(
        self,
        frame: av.VideoFrame,
    ) -> av.VideoFrame:

        image = frame.to_ndarray(
            format="bgr24",
        )

        try:

            faces = self.detector.detect(image)

            for face in faces:

                result = predict(
                    self.model,
                    face.image,
                )

                image = draw_prediction(
                    frame=image,
                    bbox=face.bbox,
                    label=result["label"],
                    confidence=result["confidence"],
                )

        except Exception as error:

            logger.exception(error)

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# ==========================================================
# WebRTC Configuration
# ==========================================================

RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302",
            ]
        }
    ]
}


# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("Detection Settings")

show_info = st.sidebar.checkbox(
    "Show Information",
    value=True,
)

st.sidebar.markdown("---")

st.sidebar.write(
    "The webcam processes every frame using "
    "MediaPipe Face Detection and the trained "
    "TensorFlow model."
)


# ==========================================================
# Webcam Stream
# ==========================================================

webrtc_ctx = webrtc_streamer(
    key="face-mask-detector",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    video_processor_factory=VideoProcessor,
    async_processing=True,
)


# ==========================================================
# Status
# ==========================================================

if webrtc_ctx.state.playing:

    st.success("Webcam is running.")

else:

    st.info(
        "Click START to begin real-time detection."
    )


# ==========================================================
# Information
# ==========================================================

if show_info:

    st.divider()

    st.subheader("How It Works")

    st.markdown(
        """
1. Capture webcam frame.

2. Detect all faces using MediaPipe.

3. Crop every detected face.

4. Preprocess the face image.

5. Predict Mask / No Mask.

6. Draw the prediction on the frame.

7. Display the processed video.
"""
    )

    st.divider()

    st.subheader("Model")

    st.write(
        """
Model: MobileNetV2

Framework: TensorFlow / Keras

Face Detection: MediaPipe

Inference: Real-Time Webcam
"""
    )


# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "Face Mask Detection using TensorFlow, MediaPipe, OpenCV and Streamlit."
)