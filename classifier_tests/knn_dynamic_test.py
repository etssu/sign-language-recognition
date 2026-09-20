import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from experiment_utils import print_dynamic_dataset_info, print_model_results, print_test_train_split
from preprocessing import load_dynamic_data, flatten_sequences


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

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

print_test_train_split(X_train, X_test)


# KNN
train_accuracy, test_accuracy, y_pred = train_and_evaluate_knn(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=7
)

print_model_results("KNN", train_accuracy, test_accuracy,  y_pred, y_test)