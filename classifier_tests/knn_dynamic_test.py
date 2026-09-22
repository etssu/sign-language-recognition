import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from experiment_utils import print_dynamic_dataset_info, print_model_results, print_test_train_split
from preprocessing import load_dynamic_data, flatten_sequences, load_dynamic_data_by_person

DATA_FILE = "../data/landmarks.csv"

def train_and_evaluate_knn(train_x,test_x,train_y,test_y,n_neighbors=3):
    train_x = flatten_sequences(train_x)
    test_x = flatten_sequences(test_x)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    model.fit(train_x, train_y)

    y_pred = model.predict(test_x)

    train_accuracy = model.score(train_x, train_y)
    test_accuracy = accuracy_score(test_y, y_pred)

    return train_accuracy, test_accuracy, y_pred

# Load data
X, y, person_ids, seq_lengths = load_dynamic_data(DATA_FILE)

print_dynamic_dataset_info(X, y, seq_lengths)

# Train / Test split
indices = np.arange(len(X))

X_train, X_test, y_train, y_test, train_seq_len, test_seq_len = load_dynamic_data_by_person(DATA_FILE, [3,4], [2])

print_test_train_split(X_train, X_test)


# KNN
train_accuracy, test_accuracy, y_pred = train_and_evaluate_knn(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=10
)

print_model_results("KNN", train_accuracy, test_accuracy,  y_pred, y_test)