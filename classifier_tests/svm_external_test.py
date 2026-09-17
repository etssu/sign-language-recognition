import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = "../data/landmarks.csv"

df = pd.read_csv(DATA_FILE)
df = df[df["gesture_type"] == "static"]

train_df = df[df["person_id"].isin([2,3])]
test_df = df[df["person_id"] == 1]

print("\nTraining samples:", len(train_df))
print("External test samples:", len(test_df))


def get_features(dataframe):
    features = []

    for _, row in dataframe.iterrows():
        landmarks = []

        for i in range(21):
            landmarks.extend([
                row[f"x{i}"],
                row[f"y{i}"],
                row[f"z{i}"]
            ])

        features.append(landmarks)

    return features


X_train = get_features(train_df)
X_test = get_features(test_df)

y_train = train_df["gesture"].values
y_test = test_df["gesture"].values


# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train SVM
svm = SVC(
    kernel="linear",
    C=0.1,
    random_state=42
)

svm.fit(X_train_scaled, y_train)

# Train accuracy
y_train_pred = svm.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"Train accuracy: {train_accuracy:.4f}")

# External test prediction
y_pred = svm.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print("SVM RESULTS")
print("=========================")

print(f"External test accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))