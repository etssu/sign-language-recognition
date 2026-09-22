import cv2
import mediapipe as mp
import numpy as np
import matplotlib
import pandas as pd

print("Environment is ready!")
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)

df = pd.read_csv("data/landmarks.csv")

df["gesture"] = df["gesture"].replace("É ", "É")

df.to_csv("data/landmarks.csv", index=False)
print("complete")