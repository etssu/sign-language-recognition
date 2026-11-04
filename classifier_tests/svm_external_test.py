
from utils.experiment_utils import print_model_results, print_test_train_split
from preprocessing import load_static_data_by_person
from model_evaluation import train_and_evaluate_svm

DATA_FILE = "../data/landmarks.csv"


X_train, X_test, y_train, y_test = load_static_data_by_person(
    DATA_FILE,
    train_person_ids=[1,2],
    test_person_ids=[3]
)

print_test_train_split(X_train, X_test)

train_accuracy, test_accuracy, y_pred, training_time, pred_time = train_and_evaluate_svm(
    X_train,
    X_test,
    y_train,
    y_test,
    kernel="linear",
    c=0.9
)

print_model_results(
    "SVM",
    train_accuracy,
    test_accuracy,
    y_pred,
    y_test
)

print(f"Training time: {training_time}s")
print(f"Average Prediction time: {pred_time}ms")