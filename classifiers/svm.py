from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from experiment_utils import print_test_train_split, print_model_results
from preprocessing import load_static_data

# 1. Load dataset
DATA_FILE = "../data/landmarks.csv"

def train_and_evaluate_svm(
    train_x,
    test_x,
    train_y,
    test_y,
    kernel="linear",
    c=0.1
):
    scaler = StandardScaler()

    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    model = SVC(
        kernel=kernel,
        C=c,
        random_state=42
    )

    model.fit(train_x_scaled, train_y)

    y_train_pred = model.predict(train_x_scaled)
    y_pred = model.predict(test_x_scaled)

    train_accuracy = accuracy_score(train_y, y_train_pred)
    test_accuracy = accuracy_score(test_y, y_pred)

    return train_accuracy, test_accuracy, y_pred


X, y = load_static_data(DATA_FILE)

# 2. Train \ test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print_test_train_split(X_train, X_test)
train_accuracy, test_accuracy, y_pred = train_and_evaluate_svm(
    X_train,
    X_test,
    y_train,
    y_test,
    kernel="linear",
    c=0.1
)

# 7. Evaluation
print_model_results("SVM", train_accuracy, test_accuracy, y_pred, y_test)