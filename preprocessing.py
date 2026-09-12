import pandas as pd
import numpy as np

# TODO - create a dynamic load
LANDMARK_COLUMNS = []

for i in range(21):
    LANDMARK_COLUMNS.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


def normalize_row(row, num_points=21):
    # Normalizes a single frame. Returns None if the frame is invalid (scale ~ 0)
    wrist_x, wrist_y = row["x0"], row["y0"] # wrist
    mid_x, mid_y = row["x9"], row["y9"] # middle finger

    scale = np.sqrt((mid_x - wrist_x) ** 2 + (mid_y - wrist_y) ** 2) # distance between the wrist and the base of the middle finger
    if scale == 0:
        scale = 1e-6

    values = []
    for i in range(num_points):
        values.append((row[f"x{i}"] - wrist_x) / scale)
        values.append((row[f"y{i}"] - wrist_y) / scale)
        values.append(row[f"z{i}"] / scale)

    return values


def load_static_data(file_path):
    df = pd.read_csv(file_path) # df - data frame
    df = df[df["gesture_type"] == "static"].copy()

    X = np.array([normalize_row(row) for _, row in df.iterrows()])
    y = df["gesture"].values

    return X, y