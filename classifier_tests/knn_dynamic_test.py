import joblib
import pandas as pd

from utils.experiment_utils import print_model_results, print_test_train_split
from preprocessing import load_dynamic_data_by_person
from model_evaluation import train_and_evaluate_knn


DATA_FILE = "../data/landmarks.csv"

X_train, X_test, y_train, y_test, train_seq_len, test_seq_len = load_dynamic_data_by_person(DATA_FILE, [1,2], [3])

print_test_train_split(X_train, X_test)

# KNN
train_accuracy, test_accuracy, y_pred, knn_model = train_and_evaluate_knn(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=15,
    data_type="dynamic"
)
print("Test distribution:")
print(pd.Series(y_test).value_counts().sort_index())

print("\nTrain distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print_model_results("KNN", train_accuracy, test_accuracy,  y_pred, y_test)
joblib.dump(knn_model, "../trained_models/knn_dynamic_model.pkl")