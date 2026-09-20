from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from experiment_utils import print_model_results
from preprocessing import load_static_data_by_person

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


X_train, X_test, y_train, y_test = load_static_data_by_person(
    DATA_FILE,
    train_person_ids=[2, 3],
    test_person_ids=[1]
)

print("\nTraining samples:", len(X_train))
print("External test samples:", len(X_test))

train_accuracy, test_accuracy, y_pred = train_and_evaluate_svm(
    X_train,
    X_test,
    y_train,
    y_test,
    kernel="linear",
    c=0.1
)

print_model_results(
    "SVM",
    train_accuracy,
    test_accuracy,
    y_pred,
    y_test
)