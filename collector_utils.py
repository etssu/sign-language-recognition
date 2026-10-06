import cv2
import time
import mediapipe as mp

webcam = cv2.VideoCapture(0)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

frame_timestamp = 0

def draw_landmarks(frame, hand_landmarks):
    # Draws hand landmarks and connections on the frame
    height, width, _ = frame.shape
    points = []

    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    # draw lines
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    # place points
    for x, y in points:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

def get_camera_frame():
    # ret(boolean) returns true if the frame is available
    # frame is an image array vector captured based on the default frames per second
    ret, frame = webcam.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    return frame

def show_countdown():
    # Shows a 3-2-1 countdown on the webcam feed. Returns False if user quits
    for number in [3, 2, 1]:
        start_time = time.time() # get actual time

        while time.time() - start_time < 1:
            frame = get_camera_frame()
            if frame is None:
                print("Camera Error.")
                return False

            cv2.putText(
                frame, str(number), (250, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10
            )

            cv2.imshow("Data Collector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("You quit.")
                return False

    return True

def wait_for_space(person_id, gesture_type, gesture):
    # Waits until SPACE is pressed. Returns False if user quits
    while True:
        frame = get_camera_frame()
        if frame is None:
            print("Camera Error.")
            return False

        cv2.putText(frame, f"Person: {person_id}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (189, 64, 137), 2)
        cv2.putText(frame, f"Gesture: {gesture} ({gesture_type})", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (189, 64, 137), 2)
        cv2.putText(frame, "Press SPACE to start", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 26, 164), 2)

        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        # user is ready
        if key == ord(" "):
            return True
        # quit
        if key == ord("q"):
            print("You quit.")
            return False

def wait_for_next_gesture():
    while True:
        frame = get_camera_frame()
        if frame is None:
            print("Camera Error.")
            return False

        cv2.putText(frame, "Press N for next gesture / Q to quit", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 26, 164), 2)
        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        # next gesture
        if key == ord("n"):
            return True
        # quit
        if key == ord("q"):
            print("You quit.")
            return False

# Static gesture collection
def collect_static_gesture(hand_landmarker, writer, file, samples_per_gesture,  sample_interval, person_id, gesture):
    global frame_timestamp

    session_id = time.strftime("%Y%m%d_%H%M%S")

    samples_collected = 0
    last_sample_time = 0

    print("COLLECTING...")

    while samples_collected < samples_per_gesture:
        frame = get_camera_frame()
        if frame is None:
            print("Camera Error.")
            return False

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = hand_landmarker.detect_for_video(mp_image, frame_timestamp)
        frame_timestamp += 1

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)

            current_time = time.time()

            if current_time - last_sample_time >= sample_interval:
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

                print(f"Saved sample {samples_collected}/{samples_per_gesture}")

        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {samples_collected}/{samples_per_gesture}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Data Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("You quit.")
            return False

    print(f"Collected: {samples_collected}")
    return True

# Dynamic gesture collection
def collect_dynamic_gesture(hand_landmarker, writer, file, recording_duration, repetitions, repetition_index, person_id, gesture):
    # Records one continuous repetition of a moving gesture
    global frame_timestamp

    session_id = time.strftime("%Y%m%d_%H%M%S") + f"_{repetition_index}"
    frame_number = 0
    start_time = time.time()

    while time.time() - start_time < recording_duration:
        frame = get_camera_frame()
        if frame is None:
            print("Camera Error.")
            return False

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
        cv2.putText(frame, f"Repetition: {repetition_index + 1}/{repetitions}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Frames recorded: {frame_number}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Data Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("You quit.")
            return False
    print(f"Repetition {repetition_index + 1}/{repetitions} done, frames: {frame_number}")

    return True

def close_camera():
    webcam.release()