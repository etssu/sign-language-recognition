import pandas as pd
import numpy as np

LANDMARK_COLUMNS = []

for i in range(21):
    LANDMARK_COLUMNS.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


def normalize_row(row, num_points=21, mirror=False):
    # Normalizes a single frame. Returns None if the frame is invalid (scale ~ 0)
    wrist_x, wrist_y = row["x0"], row["y0"] # wrist
    mid_x, mid_y = row["x9"], row["y9"] # middle finger

    scale = np.sqrt((mid_x - wrist_x) ** 2 + (mid_y - wrist_y) ** 2) # distance between the wrist and the base of the middle finger
    if scale == 0:
        scale = 1e-6

    values = []
    for i in range(num_points):
        x = (row[f"x{i}"] - wrist_x) / scale

        if mirror:
            x = -x
        
        values.append(x)
        values.append((row[f"y{i}"] - wrist_y) / scale)
        values.append(row[f"z{i}"] / scale)

    return values


def load_static_data(file_path):
    df = pd.read_csv(file_path) # df - data frame
    df = df[df["gesture_type"] == "static"].copy()

    X = np.array([normalize_row(row) for _, row in df.iterrows()])
    y = df["gesture"].values

    return X, y

def load_dynamic_data(file_path, max_seq_len=None):
    df = pd.read_csv(file_path)
    df = df[df["gesture_type"] == "dynamic"].copy()

    sequences = []
    labels = []
    person_ids = []

    # group by person + session = one gesture record
    grouped = df.groupby(["person_id", "session_id"])

    for (person_id, session_id), group in grouped:
        group = group.sort_values("frame_number")

        # normalize every frame separately
        frames = np.array([normalize_row(row) for _, row in group.iterrows()])

        sequences.append(frames)
        labels.append(group["gesture"].iloc[0])
        person_ids.append(person_id)

    # statistics about sequence lengths
    lengths = [len(seq) for seq in sequences]

    print("Number of sequences:", len(sequences))
    print("Min sequence length:", min(lengths))
    print("Max sequence length:", max(lengths))
    print("Average sequence length:", sum(lengths) / len(lengths))

    if max_seq_len is None:
        max_seq_len = max(len(seq) for seq in sequences)

    n_features = sequences[0].shape[1]
    X = np.zeros((len(sequences), max_seq_len, n_features), dtype=np.float32)
    seq_lengths = np.zeros(len(sequences), dtype=np.int32)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_seq_len)
        X[i, :length] = seq[:length]
        seq_lengths[i] = length

    y = np.array(labels)
    person_ids = np.array(person_ids)

    return X, y, person_ids, seq_lengths

def flatten_sequences(X):
    return X.reshape(X.shape[0], -1)