import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import load_static_data

# 1. Load dataset
DATA_FILE = "../data/landmarks.csv"

X, y = load_static_data(DATA_FILE)

# 3. Train \ test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the classifier
svm_classifier = SVC(kernel='rbf', C=0.1, random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

# calculate & print train accuracy
y_train_pred = svm_classifier.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"Train accuracy: {train_accuracy:.4f}")

# 6. Prediction
y_pred = svm_classifier.predict(X_test_scaled)

# 7. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print("SVM RESULTS")
print("=========================")

print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))