import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
import time


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]


# Settings

# Static gestures
SAMPLES_PER_GESTURE = 50
SAMPLE_INTERVAL = 0.15

# Dynamic gestures
RECORDING_DURATION = 1.5   # seconds per repetition
REPETITIONS = 15           # number of repetitions to record

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "landmarks.csv")

os.makedirs(DATA_DIR, exist_ok=True)


# User information
person_id = input("Enter person ID: ")
gesture = input("Enter gesture: ")

gesture_type = input("Gesture type (static/dynamic): ").strip().lower()

while gesture_type not in ("static", "dynamic"):
    gesture_type = input("Please type 'static' or 'dynamic': ").strip().lower()


# MediaPipe
base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)


# CSV
file_exists = os.path.exists(DATA_FILE)

file = open(DATA_FILE, "a", newline="")
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


# Webcam
webcam = cv2.VideoCapture(0)

frame_timestamp = 0


def draw_landmarks(frame, hand_landmarks):
    # Draws hand landmarks and connections on the frame
    h, w, c = frame.shape
    points = []

    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


def show_countdown():
    # Shows a 3-2-1 countdown on the webcam feed. Returns False if user quits
    for number in [3, 2, 1]:
        start_time = time.time()

        while time.time() - start_time < 1:
            ret, frame = webcam.read()
            if not ret:
                return False

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame, str(number), (250, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10
            )

            cv2.imshow("Data Collector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

    return True


def wait_for_space():
    # Waits until SPACE is pressed. Returns False if user quits
    while True:
        ret, frame = webcam.read()
        if not ret:
            return False

        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f"Person: {person_id}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture} ({gesture_type})", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to start", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            return True

        if key == ord("q"):
            return False


def cleanup_and_exit():
    webcam.release()
    file.close()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    exit()


# Static gesture collection

def collect_static_gesture():
    global frame_timestamp

    session_id = time.strftime("%Y%m%d_%H%M%S")

    samples_collected = 0
    last_sample_time = 0

    print("COLLECTING...")

    while samples_collected < SAMPLES_PER_GESTURE:

        ret, frame = webcam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = hand_landmarker.detect_for_video(mp_image, frame_timestamp)
        frame_timestamp += 1

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)

            current_time = time.time()

            if current_time - last_sample_time >= SAMPLE_INTERVAL:
                row = [
                    person_id,
                    session_id,
                    gesture,
                    "static",
                    samples_collected
                ]

                for landmark in hand_landmarks:
                    row.extend([landmark.x, landmark.y, landmark.z])

                writer.writerow(row)
                file.flush()

                samples_collected += 1
                last_sample_time = current_time

                print(f"Saved sample {samples_collected}/{SAMPLES_PER_GESTURE}")

        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {samples_collected}/{SAMPLES_PER_GESTURE}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Data Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Collected: {samples_collected}")



# Dynamic gesture collection

def collect_dynamic_gesture(repetition_index):
    # Records one continuous repetition of a moving gesture
    global frame_timestamp

    session_id = time.strftime("%Y%m%d_%H%M%S") + f"_{repetition_index}"
    frame_number = 0
    start_time = time.time()

    while time.time() - start_time < RECORDING_DURATION:

        ret, frame = webcam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = hand_landmarker.detect_for_video(mp_image, frame_timestamp)
        frame_timestamp += 1

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)

            row = [
                person_id,
                session_id,
                gesture,
                "dynamic",
                frame_number
            ]

            for landmark in hand_landmarks:
                row.extend([landmark.x, landmark.y, landmark.z])

            writer.writerow(row)
            file.flush()

            frame_number += 1

        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Repetition: {repetition_index + 1}/{REPETITIONS}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Frames recorded: {frame_number}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Data Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    print(f"Repetition {repetition_index + 1}/{REPETITIONS} done, frames: {frame_number}")
    return True



# Main flow
print()
print("================================")
print("DATA COLLECTION")
print("================================")
print(f"Person: {person_id}")
print(f"Gesture: {gesture}")
print(f"Type: {gesture_type}")

if gesture_type == "static":
    print(f"Samples: {SAMPLES_PER_GESTURE}")
else:
    print(f"Repetitions: {REPETITIONS}, duration each: {RECORDING_DURATION}s")

print()
print("Press SPACE to start")
print("Press Q to quit")
print("================================")

if not wait_for_space():
    cleanup_and_exit()

if gesture_type == "static":
    if not show_countdown():
        cleanup_and_exit()

    collect_static_gesture()

else:  # dynamic
    for rep in range(REPETITIONS):

        print(f"Repetition {rep + 1}/{REPETITIONS} - get ready")

        if not show_countdown():
            break

        print("GO - perform the gesture now")

        success = collect_dynamic_gesture(rep)

        if not success:
            break


# Cleanup
webcam.release()
file.close()
cv2.destroyAllWindows()
hand_landmarker.close()

print()
print("================================")
print("COLLECTION FINISHED")
print(f"Saved to: {DATA_FILE}")
print("================================")