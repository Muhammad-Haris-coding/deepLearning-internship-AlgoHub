import logging
import time

import cv2
import streamlit as st

from face_processor import FaceProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Real-Time Face Recognition",
    page_icon="👤",
    layout="wide"
)

st.title("👤 Real-Time Face Recognition")
st.write("Live Face Recognition using OpenCV + face_recognition")

try:
    processor = FaceProcessor("known_faces")
except Exception as exc:
    logger.exception("Failed to initialize face processor")
    st.error(f"Unable to load known faces: {exc}")
    st.stop()

run = st.checkbox("Start Camera")
frame_window = st.empty()
status = st.empty()

if "cap" not in st.session_state:
    st.session_state.cap = None

if not run:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.info("Click 'Start Camera' to begin.")
    st.stop()

if st.session_state.cap is None:
    camera_error = None
    for index in [0, 1, 2]:
        temp = cv2.VideoCapture(index)
        if temp.isOpened():
            st.session_state.cap = temp
            logger.info("Opened camera at index %s", index)
            break
        temp.release()
        camera_error = (
            f"Unable to open camera index {index}. The camera may be locked by another application "
            "or the selected index may be incorrect."
        )

    if st.session_state.cap is None:
        st.error(camera_error or "No webcam found.")
        st.stop()

cap = st.session_state.cap
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

success, frame = cap.read()
if not success:
    logger.error("Failed to read frame from camera")
    st.error("The webcam frame could not be read. The camera may have disconnected or become unavailable.")
    cap.release()
    st.session_state.cap = None
    st.stop()

faces = processor.recognize(frame)

for face in faces:
    top, right, bottom, left = face["box"]
    name = face["name"]
    color = (0, 255, 0)
    if name == "Unknown":
        color = (0, 0, 255)

    padding = 40
    top = max(0, top - padding)
    right = min(frame.shape[1], right + padding)
    bottom = min(frame.shape[0], bottom + padding)
    left = max(0, left - padding)

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        frame,
        name,
        (left + 6, bottom - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_window.image(rgb, channels="RGB", use_container_width=True)
status.info("Camera stream is active. Streamlit is updating the latest frame.")
time.sleep(0.05)
