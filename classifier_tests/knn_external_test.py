from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from experiment_utils import print_model_results
from preprocessing import load_static_data_by_person

DATA_FILE = "../data/landmarks.csv"


def train_and_evaluate_knn(train_x,test_x,train_y,test_y,n_neighbors=7):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    model.fit(train_x, train_y)

    y_train_pred = model.predict(train_x)
    y_pred = model.predict(test_x)

    train_accuracy = accuracy_score(
        train_y,
        y_train_pred
    )

    test_accuracy = accuracy_score(
        test_y,
        y_pred
    )

    return train_accuracy, test_accuracy, y_pred


# Load data
X_train, X_test, y_train, y_test = load_static_data_by_person(
    DATA_FILE,
    train_person_ids=[2, 3],
    test_person_ids=[1]
)

print("\nTraining samples:", len(X_train))
print("External test samples:", len(X_test))


# KNN
train_accuracy, test_accuracy, y_pred = train_and_evaluate_knn(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=7
)

print_model_results(
    "KNN",
    train_accuracy,
    test_accuracy,
    y_test,
    y_pred
)
