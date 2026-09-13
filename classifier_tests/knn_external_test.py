import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_FILE = "data/landmarks.csv"


# 1. Load data
df = pd.read_csv(DATA_FILE)

# Only static gestures
df = df[df["gesture_type"] == "static"]


# 2. Split by person
train_df = df[df["person_id"] == 1]
test_df = df[df["person_id"] == 2]

print("\nTraining samples:", len(train_df))
print("External test samples:", len(test_df))


# 3. Normalization
def normalize_landmarks(dataframe):

    normalized_data = []

    for _, row in dataframe.iterrows():

        landmarks = []

        for i in range(21):
            landmarks.append([
                row[f"x{i}"],
                row[f"y{i}"],
                row[f"z{i}"]
            ])

        landmarks = np.array(landmarks)

        # Move wrist (landmark 0) to origin
        landmarks = landmarks - landmarks[0]

        # Scale according to hand size
        max_distance = np.max(np.linalg.norm(landmarks, axis=1))

        if max_distance != 0:
            landmarks = landmarks / max_distance

        normalized_data.append(landmarks.flatten())

    return np.array(normalized_data)


X_train = normalize_landmarks(train_df)
X_test = normalize_landmarks(test_df)

y_train = train_df["gesture"].values
y_test = test_df["gesture"].values


# 4. KNN
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)


# 5. Train accuracy
y_train_pred = knn.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)


# 6. External test
y_pred = knn.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_pred
)


# 7. Results
print("\n=========================")
print("KNN - NORMALIZED EXTERNAL TEST")
print("=========================")

print(f"Train accuracy: {train_accuracy:.4f}")
print(f"External test accuracy: {test_accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))