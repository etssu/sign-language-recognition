import numpy as np

from experiment_utils import print_dynamic_dataset_info, print_model_results, print_test_train_split
from preprocessing import load_dynamic_data, flatten_sequences, load_dynamic_data_by_person
from model_evaluation import train_and_evaluate_knn


DATA_FILE = "../data/landmarks.csv"

X_train, X_test, y_train, y_test, train_seq_len, test_seq_len = load_dynamic_data_by_person(DATA_FILE, [3,4], [2])

print_test_train_split(X_train, X_test)

# KNN
train_accuracy, test_accuracy, y_pred = train_and_evaluate_knn(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=10,
    data_type="dynamic"
)

print_model_results("KNN", train_accuracy, test_accuracy,  y_pred, y_test)