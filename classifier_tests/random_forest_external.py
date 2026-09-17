import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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


rf = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"Train accuracy: {train_accuracy:.4f}")

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print("RANDOM FOREST RESULTS")
print("=========================")

print(f"External test accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))