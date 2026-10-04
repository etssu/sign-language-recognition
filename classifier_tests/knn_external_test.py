
from experiment_utils import print_model_results, print_test_train_split
from preprocessing import load_static_data_by_person
from model_evaluation import train_and_evaluate_knn

DATA_FILE = "../data/landmarks.csv"


# Load data
X_train, X_test, y_train, y_test = load_static_data_by_person(
    DATA_FILE,
    train_person_ids=[1],
    test_person_ids=[2]
)

print_test_train_split(X_train, X_test)


# KNN - static
train_accuracy, test_accuracy, y_pred = train_and_evaluate_knn(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=7,
)

print_model_results(
    "KNN",
    train_accuracy,
    test_accuracy,
    y_pred,
    y_test
)
