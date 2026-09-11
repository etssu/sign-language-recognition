import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
DATA_FILE = "../data/landmarks.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nClasses:")
print(df["gesture"].value_counts())

# 2. Prepare data
excluded_columns = ["person_id", "session_id", "gesture"]

X = df.drop(columns=excluded_columns)
y = df["gesture"]

# 3. Train / test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Create Random Forest model
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_classifier.fit(X_train, y_train)


# 5. Train accuracy
y_train_pred = rf_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"\nTrain accuracy: {train_accuracy:.4f}")


# 6. Prediction
y_pred = rf_classifier.predict(X_test)


# 7. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print("RANDOM FOREST RESULTS")
print("=========================")

print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
