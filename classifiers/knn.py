import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import load_static_data

# 1. Load dataset
DATA_FILE = "../data/landmarks.csv"

X, y = load_static_data(DATA_FILE)

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 3. Create KNN model
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

# calculate & print train accuracy
y_train_pred = knn.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"Train accuracy: {train_accuracy:.4f}")

# 4. Prediction
y_pred = knn.predict(X_test)


# 5. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print("KNN RESULTS")
print("=========================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))