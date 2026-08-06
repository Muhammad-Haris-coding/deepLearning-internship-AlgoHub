import logging
import os
from pathlib import Path

import cv2
import face_recognition
import numpy as np


class FaceProcessor:
    def __init__(self, known_faces_dir="known_faces", tolerance=0.5):
        self.logger = logging.getLogger(__name__)
        self.tolerance = tolerance
        self.known_faces_dir = Path(known_faces_dir)

        if not self.known_faces_dir.is_absolute():
            self.known_faces_dir = Path(__file__).resolve().parent / self.known_faces_dir

        self.known_encodings = []
        self.known_names = []

        self.load_known_faces()

    def load_known_faces(self):
        if not self.known_faces_dir.exists():
            raise FileNotFoundError(
                f"Folder '{self.known_faces_dir}' does not exist."
            )

        supported = (".jpg", ".jpeg", ".png")

        for file in sorted(self.known_faces_dir.iterdir()):
            if not file.is_file() or not file.name.lower().endswith(supported):
                continue

            try:
                image = face_recognition.load_image_file(str(file))
                encodings = face_recognition.face_encodings(image)
            except Exception as exc:
                self.logger.exception("Failed to load/encode %s", file.name)
                continue

            if len(encodings) == 0:
                self.logger.warning("No face found in %s", file.name)
                continue

            self.known_encodings.append(encodings[0])

            name_map = {
                "my_photo.jpeg": "Muhammad Haris",
                "my_photo.jpg": "Muhammad Haris",
            }
            name = name_map.get(file.name.lower(), file.stem.replace("_", " ").title())
            self.known_names.append(name)

        if len(self.known_encodings) == 0:
            raise ValueError(
                "No valid face encodings were found inside known_faces folder."
            )

    def recognize(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            locations = face_recognition.face_locations(small)
            encodings = face_recognition.face_encodings(small, locations)
        except Exception as exc:
            self.logger.exception("Face recognition processing failed")
            return []

        results = []

        for (top, right, bottom, left), encoding in zip(locations, encodings):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            distances = face_recognition.face_distance(
                self.known_encodings,
                encoding
            )

            name = "Unknown"

            if len(distances) > 0:
                best = np.argmin(distances)
                if distances[best] < self.tolerance:
                    name = self.known_names[best]

            results.append(
                {
                    "name": name,
                    "box": (top, right, bottom, left)
                }
            )

        return results