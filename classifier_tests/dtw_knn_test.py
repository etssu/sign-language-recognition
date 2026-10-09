from math import inf

import pandas as pd
import numpy as np
from dtaidistance import dtw_ndim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from experiment_utils import print_test_train_split
from preprocessing import load_dynamic_data_for_dtw


DATA_FILE = "../data/landmarks.csv"

df = pd.read_csv(DATA_FILE)

X_train, X_test, y_train, y_test = load_dynamic_data_for_dtw(DATA_FILE, [1,2], [3])

selected_gestures = {"Á", "Ä", "É", "Č", "Ď"}

train_mask = np.isin(y_train, list(selected_gestures))
test_mask = np.isin(y_test, list(selected_gestures))

X_train = [seq for seq, keep in zip(X_train, train_mask) if keep]
y_train = y_train[train_mask]

X_test = [seq for seq, keep in zip(X_test, test_mask) if keep]
y_test = y_test[test_mask]

y_pred = []
k = 7

for test_sample in X_test:
    distances = []

    for train_sample, train_label in zip(X_train, y_train):
        distance = dtw_ndim.distance_fast(
            test_sample,
            train_sample
        )
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

y_train_pred = []
k = 7

for i, train_sample in enumerate(X_train):
    distances = []

    for j, (tr_sample, train_label) in enumerate(zip(X_train, y_train)):
        if i == j:
            continue

        distance = dtw_ndim.distance_fast(train_sample, tr_sample)
        distances.append((distance, train_label))

    distances.sort(key=lambda item: item[0])
    nearest_neighbours = distances[:k]
    neighbour_labels = [label for distance, label in nearest_neighbours]

    predicted_label = max(set(neighbour_labels), key=neighbour_labels.count)
    y_train_pred.append(predicted_label)

print("Train Accuracy: ", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=sorted(set(y_test))))