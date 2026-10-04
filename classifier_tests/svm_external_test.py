
from experiment_utils import print_model_results
from preprocessing import load_static_data_by_person
from model_evaluation import train_and_evaluate_svm

DATA_FILE = "../data/landmarks.csv"


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