
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from dtaidistance import dtw_ndim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from time import perf_counter

from preprocessing import load_dynamic_data_by_person


DATA_FILE = "../data/landmarks.csv"

TRAINING_SIZES = [120, 240, 360, 600, 900, 1200, 1439]

PREDICTION_TIMES = []
AVG_PREDICTION_TIMES = []

k = 7


# Load dynamic gesture data
X_train, X_test, y_train, y_test, train_seq_len, test_seq_len = load_dynamic_data_by_person(
    DATA_FILE, [1, 2], [3]
)

# Convert sequences to double precision
X_train = [np.asarray(sample, dtype=np.double) for sample in X_train]
X_test = [np.asarray(sample, dtype=np.double) for sample in X_test]

# Remove unnecessary spaces from labels
y_train = np.array([label.strip() for label in y_train])
y_test = np.array([label.strip() for label in y_test])

print("Training sequences:", len(X_train))
print("Testing sequences:", len(X_test))
print("Training distribution:", Counter(y_train))


# Test prediction with different training dataset sizes
for num_samples in TRAINING_SIZES:

    # Select a stratified subset of training data
    if num_samples == len(X_train):
        X_train_subset = X_train
        y_train_subset = y_train
    else:
        indices = np.arange(len(X_train))

        subset_indices, _ = train_test_split(
            indices,
            train_size=num_samples,
            stratify=y_train,
            random_state=42
        )

        X_train_subset = [X_train[i] for i in subset_indices]
        y_train_subset = y_train[subset_indices]

    y_pred_subset = []

    # Measure prediction time
    start_time_pred = perf_counter()

    # Predict gestures for test sequences
    for test_sample in X_test:
        distances = []

        # Calculate DTW distance to each training sequence
        for train_sample, train_label in zip(X_train_subset, y_train_subset):
            distance = dtw_ndim.distance_fast(test_sample, train_sample)
            distances.append((distance, train_label))

        # Sort neighbours by distance
        distances.sort(key=lambda item: item[0])

        # Select k nearest neighbours
        nearest_neighbours = distances[:k]

        # Collect their labels
        neighbour_labels = [label for distance, label in nearest_neighbours]

        # Predict the most common label
        predicted_label = max(set(neighbour_labels), key=neighbour_labels.count)
        y_pred_subset.append(predicted_label)

    # Calculate total and average prediction time
    pred_time = perf_counter() - start_time_pred
    avg_prediction_time = pred_time / len(X_test) * 1000

    PREDICTION_TIMES.append(pred_time)
    AVG_PREDICTION_TIMES.append(avg_prediction_time)

    print(f"\nTraining samples: {num_samples}")
    print(f"Total prediction time: {pred_time:.4f} s")
    print(f"Average prediction time: {avg_prediction_time:.4f} ms")


# Train prediction for accuracy evaluation
y_train_pred = []

# Leave-one-out prediction on training data
for i, train_sample in enumerate(X_train):
    distances = []

    for j, (tr_sample, train_label) in enumerate(zip(X_train, y_train)):

        # Skip the current sample
        if i == j:
            continue

        # Calculate DTW distance
        distance = dtw_ndim.distance_fast(train_sample, tr_sample)
        distances.append((distance, train_label))

    # Sort neighbours by distance
    distances.sort(key=lambda item: item[0])

    # Select k nearest neighbours
    nearest_neighbours = distances[:k]

    # Collect their labels
    neighbour_labels = [label for distance, label in nearest_neighbours]

    # Predict the most common label
    predicted_label = max(set(neighbour_labels), key=neighbour_labels.count)
    y_train_pred.append(predicted_label)


# Test prediction on the full training dataset
y_pred = []

for test_sample in X_test:
    distances = []

    for train_sample, train_label in zip(X_train, y_train):
        distance = dtw_ndim.distance_fast(test_sample, train_sample)
        distances.append((distance, train_label))

    # Sort neighbours by distance
    distances.sort(key=lambda item: item[0])

    # Select k nearest neighbours
    nearest_neighbours = distances[:k]

    # Collect their labels
    neighbour_labels = [label for distance, label in nearest_neighbours]

    # Predict the most common label
    predicted_label = max(set(neighbour_labels), key=neighbour_labels.count)
    y_pred.append(predicted_label)


# Evaluate classifier
print("\nTrain Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=sorted(set(y_test))))


# Plot prediction time depending on training dataset size
plt.figure(figsize=(9, 5))
plt.plot(TRAINING_SIZES, AVG_PREDICTION_TIMES, marker="o")

plt.xlabel("Number of training sequences")
plt.ylabel("Average prediction time (ms/sequence)")
plt.title("DTW-KNN prediction time depending on training dataset size")

plt.grid(True)
plt.tight_layout()
plt.show()
