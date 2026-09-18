import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import load_dynamic_data


DATA_FILE = "../data/landmarks.csv"
X, y, person_ids, seq_lengths = load_dynamic_data(DATA_FILE)

print("\n=========================")
print("DYNAMIC DATA")
print("=========================")

print("Number of sequences:", len(X))
print("X shape:", X.shape)
print("Classes:", np.unique(y))
print("Number of classes:", len(np.unique(y)))

print("\nSequence lengths:")
print("Min:", seq_lengths.min())
print("Max:", seq_lengths.max())
print("Average:", seq_lengths.mean())

print("\nSamples per gesture:")
for gesture in np.unique(y):
    print(gesture, ":", np.sum(y == gesture))


# TRAIN / TEST SPLIT
indices = np.arange(len(X))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]


print("\n=========================")
print("TRAIN / TEST SPLIT")
print("=========================")

print("Training sequences:", len(X_train))
print("Testing sequences:", len(X_test))



# FLATTEN SEQUENCES
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print("\nFlattened X_train shape:", X_train.shape)
print("Flattened X_test shape:", X_test.shape)


# KNN
model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# RESULTS
train_accuracy = model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print("KNN RESULTS")
print("=========================")

print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))