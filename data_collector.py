import cv2
import os
import csv

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from collector_utils import wait_for_space, show_countdown, collect_static_gesture, collect_dynamic_gesture, \
    wait_for_next_gesture, close_camera


# SETTINGS
# Static gestures
SAMPLES_PER_GESTURE = 50
SAMPLE_INTERVAL = 0.15

# Dynamic gestures
RECORDING_DURATION = 1.5   # seconds per repetition
REPETITIONS = 15           # number of repetitions to record

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "landmarks.csv")

# MediaPipe settings
base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# CSV
file_exists = os.path.exists(DATA_FILE)

file = open(DATA_FILE, "a", newline="", encoding="utf-8")
writer = csv.writer(file)

if not file_exists:
    header = [
        "person_id",
        "session_id",
        "gesture",
        "gesture_type",
        "frame_number"
    ]

    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])

    writer.writerow(header)


person_id = input("Enter person ID: ")
gesture_type = input("Enter gesture type(static/dynamic): ")

while gesture_type not in ("static", "dynamic"):
    gesture_type = input("Please type 'static' or 'dynamic': ")
while True:
    gesture = input("Enter gesture: ")
    if gesture.lower() == 'quit':
        break

    if not wait_for_space(person_id, gesture_type, gesture):
        break

    success = True

    if gesture_type == "static":
        if not show_countdown():
            success = False
        elif not collect_static_gesture(hand_landmarker, writer, file, SAMPLES_PER_GESTURE, SAMPLE_INTERVAL, person_id, gesture):
            success = False
    else:
        for repetition_index in range(REPETITIONS):
            if not show_countdown():
                success = False
                break
            if not collect_dynamic_gesture(hand_landmarker, writer, file, RECORDING_DURATION, REPETITIONS, repetition_index, person_id, gesture):
                success = False
                break

    if not success:
        break

    if not wait_for_next_gesture():
        break



# Cleanup
close_camera()
file.close()
cv2.destroyAllWindows()
hand_landmarker.close()

