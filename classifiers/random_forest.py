import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import load_static_data

# 1. Load dataset
DATA_FILE = "../data/landmarks.csv"

X, y = load_static_data(DATA_FILE)

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Create Random Forest model
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_classifier.fit(X_train, y_train)


# 4. Train accuracy
y_train_pred = rf_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"\nTrain accuracy: {train_accuracy:.4f}")


# 5. Prediction
y_pred = rf_classifier.predict(X_test)


# 6. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print("RANDOM FOREST RESULTS")
print("=========================")

print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
