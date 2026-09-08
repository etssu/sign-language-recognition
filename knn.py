import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# 1. Load dataset
DATA_FILE = "data/landmarks.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nClasses:")
print(df["gesture"].value_counts())


# 2. Prepare data

# These columns are not features
excluded_columns = ["person_id", "session_id", "gesture"]

X = df.drop(columns=excluded_columns)
y = df["gesture"]



# 3. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))



# 4. Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)



# 5. Prediction
y_pred = knn.predict(X_test)



# 6. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print("KNN RESULTS")
print("=========================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))