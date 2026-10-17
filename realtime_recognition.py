from math import sqrt

import cv2
import mediapipe as mp
import numpy as np
import time

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from model_evaluation import train_knn
from preprocessing import load_static_data_by_person, load_dynamic_data_by_person, normalize_landmarks, flatten_sequences

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

DATA_FILE = "data/landmarks.csv"
SEQUENCE_LENGTH = 46

X_train_static, X_test_static, y_train_static, y_test_static = load_static_data_by_person(DATA_FILE, train_person_ids=[2,3,5], test_person_ids=[1])
model_static = train_knn(X_train_static, y_train_static)

X_train_dynamic, X_test_dynamic, y_train_dynamic, y_test_dynamic, train_seq_lengths, test_seq_lengths = load_dynamic_data_by_person(DATA_FILE, train_person_ids=[3], test_person_ids=[1])
X_train_dynamic = flatten_sequences(X_train_dynamic)
X_test_dynamic = flatten_sequences(X_test_dynamic)
model_dynamic = train_knn(X_train_dynamic, y_train_dynamic)

# MediaPipe
base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)
# Initialize webcam and the sliding buffer
webcam = cv2.VideoCapture(0)
buffer= deque(maxlen=SEQUENCE_LENGTH)
prediction = None


def draw_landmarks(frame, hand_landmarks):
    # Draws hand landmarks and connections on the frame
    h, w, _ = frame.shape
    points = []

    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


try:
    last_timestamp_ms = -1
    previous_wrist = None

    while True:
        ret, frame = webcam.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # MediaPipe expects an RGB image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(time.time() * 1000) # get current ms

        if (timestamp_ms <= last_timestamp_ms):
            last_timestamp_ms += 1
        else:
            last_timestamp_ms = timestamp_ms


        result = hand_landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            current_wrist = (landmarks[0].x, landmarks[0].y)
            distance = 0

            if previous_wrist is None:
                print("None")
            else:
                dx = current_wrist[0] - previous_wrist[0]
                dy = current_wrist[1] - previous_wrist[1]
                distance = sqrt(dx**2 + dy**2)

            previous_wrist = current_wrist
            cv2.putText(
                frame,
                f"Movement: {distance:.4f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            draw_landmarks(frame, landmarks)

            features = normalize_landmarks(landmarks)
            buffer.append(features)

            if len(buffer) == SEQUENCE_LENGTH:
                # Convert the complete sequence into one feature vector
                sequence = np.asarray(buffer).flatten()


                if sequence.size == X_train_dynamic.shape[1]:
                    prediction = model_dynamic.predict([sequence])[0]
                else:
                    prediction = None
                    cv2.putText(
                        frame,
                        "Sequence size mismatch",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
        else:
            # Wait for a new complete sequence when the hand disappears
            previous_wrist = None
            buffer.clear()
            prediction = None

        if prediction is not None:
            cv2.putText(frame,f"Prediction: {prediction}",(20, 50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)

        cv2.imshow("Dynamic Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    webcam.release()
    hand_landmarker.close()
    cv2.destroyAllWindows()
