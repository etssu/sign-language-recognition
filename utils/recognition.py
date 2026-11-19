import time
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from preprocessing import normalize_landmarks

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
            features = normalize_landmarks(result.hand_landmarks[0])
            prediction = self.static_model.predict([features])[0]

            return prediction

        return None

    def close(self):
        self.hand_landmarker.close()
