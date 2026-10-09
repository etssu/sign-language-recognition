import numpy as np

from experiment_utils import print_dynamic_dataset_info, print_model_results, print_test_train_split
from preprocessing import load_dynamic_data, flatten_sequences, load_dynamic_data_by_person
from model_evaluation import train_and_evaluate_knn


DATA_FILE = "../data/landmarks.csv"

X_train, X_test, y_train, y_test, train_seq_len, test_seq_len = load_dynamic_data_by_person(DATA_FILE, [1,2], [3])
selected_gestures = {"Á", "Ä", "É", "Č", "Ď"}

train_mask = np.isin(y_train, list(selected_gestures))
test_mask = np.isin(y_test, list(selected_gestures))

X_train = [seq for seq, keep in zip(X_train, train_mask) if keep]
y_train = y_train[train_mask]

X_test = [seq for seq, keep in zip(X_test, test_mask) if keep]
y_test = y_test[test_mask]

X_train = np.array(X_train)
X_test = np.array(X_test)
print_test_train_split(X_train, X_test)

# KNN
train_accuracy, test_accuracy, y_pred = train_and_evaluate_knn(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=7,
    data_type="dynamic"
)

print_model_results("KNN", train_accuracy, test_accuracy,  y_pred, y_test)