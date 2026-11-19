import time
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from preprocessing import normalize_landmarks

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

class Recognition:
    def __init__(self, static_model, dynamic_model):
        self.static_model = static_model
        self.dynamic_model = dynamic_model

        model_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "hand_landmarker.task"
        )

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            options
        )

        self.last_timestamp_ms = -1

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(time.time() * 1000)

        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1

        self.last_timestamp_ms = timestamp_ms

        result = self.hand_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )
        if result.hand_landmarks:
            self.draw_landmarks(frame, result.hand_landmarks[0])
            features = normalize_landmarks(result.hand_landmarks[0])
            prediction = self.static_model.predict([features])[0]

            return prediction

        return None

    def draw_landmarks(self, frame, landmarks):
        height, width, _ = frame.shape

        points = []

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )

        for x, y in points:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    def close(self):
        self.hand_landmarker.close()
