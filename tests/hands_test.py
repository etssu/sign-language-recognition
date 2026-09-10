import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), # big thumb
    (0,5), (5,6), (6,7), (7,8), # index
    (5,9), (9,10), (10,11), (11,12), # middle
    (9,13), (13,14), (14,15), (15,16), # ring
    (13,17), (0,17), (17,18), (18,19), (19,20) # pinky
]

# Create Hand Landmarker
base_options = python.BaseOptions(model_asset_path="../models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=vision.RunningMode.VIDEO)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Open a Webcam
webcam = cv2.VideoCapture(0)

frame_timestamp = 0

while True:
    ret, frame = webcam.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect Hands
    result = hand_landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    # Draw landmarks
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, c = frame.shape

            points = []

            # landmarks -> coordinates
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append((x, y))

            # Green lines
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame,points[start],points[end],(0, 255, 0),2)

            # Red points
            for x, y in points:
                cv2.circle(frame,(x, y),5,(0, 0, 255),-1)

            for i, landmark in enumerate(hand_landmarks):
                print(i, landmark.x, landmark.y, landmark.z)



    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()