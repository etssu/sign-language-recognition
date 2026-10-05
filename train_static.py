import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from model_evaluation import train_knn
from preprocessing import load_static_data_by_person, normalize_landmarks

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

DATA_FILE = "data/landmarks.csv"

X_train, X_test, y_train, y_test = load_static_data_by_person(DATA_FILE, train_person_ids=[2,3], test_person_ids=[1])

model = train_knn(X_train, y_train, 7)

# MediaPipe
base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

webcam = cv2.VideoCapture(0)

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



while True:
    ret, frame = webcam.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    timestamp_ms = int(
        cv2.getTickCount() / cv2.getTickFrequency() * 1000
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    result = hand_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        draw_landmarks(frame, landmarks)

        features = normalize_landmarks(landmarks)

        prediction = model.predict([features])[0]

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
