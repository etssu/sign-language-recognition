import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)


while True:
    ret, frame = webcam.read()
    if ret:
        cv2.imshow("1st Frame", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cv2.destroyAllWindows()