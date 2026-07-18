"""
Face Detector
-------------
Detects faces using MediaPipe Face Detection.

Author : Afaq Ahmad Khan
Project: Face Mask Detection
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class Face:
    """
    Represents a detected face.

    Attributes
    ----------
    bbox : Tuple[int, int, int, int]
        Bounding box (x, y, w, h).

    image : np.ndarray
        Cropped RGB face image.
    """

    bbox: Tuple[int, int, int, int]
    image: np.ndarray


class FaceDetector:
    """
    MediaPipe Face Detector.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
    ):

        self.mp_face_detection = mp.solutions.face_detection

        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(
        self,
        frame: np.ndarray,
    ) -> List[Face]:
        """
        Detect faces in a BGR image.

        Parameters
        ----------
        frame : np.ndarray
            OpenCV image (BGR).

        Returns
        -------
        List[Face]
        """

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = self.detector.process(rgb)

        faces = []

        if not results.detections:
            return faces

        height, width = frame.shape[:2]

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = max(0, int(bbox.xmin * width))
            y = max(0, int(bbox.ymin * height))
            w = int(bbox.width * width)
            h = int(bbox.height * height)

            x2 = min(width, x + w)
            y2 = min(height, y + h)

            face = rgb[y:y2, x:x2]

            if face.size == 0:
                continue

            faces.append(
                Face(
                    bbox=(x, y, w, h),
                    image=face,
                )
            )

        return faces

    def close(self):
        """
        Release MediaPipe resources.
        """
        self.detector.close()


if __name__ == "__main__":

    detector = FaceDetector()

    image = cv2.imread("sample.jpg")

    if image is None:
        raise FileNotFoundError("sample.jpg not found.")

    faces = detector.detect(image)

    print(f"Detected Faces: {len(faces)}")

    for face in faces:

        x, y, w, h = face.bbox

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detector.close()