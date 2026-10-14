import cv2
import mediapipe as mp
import numpy as np
import matplotlib
import pandas as pd

print("Environment is ready!")
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)

df = pd.read_csv("data/landmarks.csv")

person_id = 5  # ID девочки

person_data = df[df["person_id"] == person_id].copy()

print(df[df["gesture"].isna()].shape)
print(df[df["gesture"].isna()][
    ["person_id", "session_id", "gesture", "frame_number"]
].to_string())

print(df[df["session_id"].astype(str).str.startswith("20261013_1038")]
      .groupby(["session_id", "gesture"], dropna=False)
      .size().to_string())

