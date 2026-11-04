from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocessing import flatten_sequences
from time import perf_counter


def train_and_evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors=10, data_type="static"):
    if data_type == 'dynamic':
        X_train = flatten_sequences(X_train)
        X_test = flatten_sequences(X_test)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    start_time_train = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - start_time_train


    y_train_pred = model.predict(X_train)

    start_time_pred = perf_counter()
    y_pred = model.predict(X_test)
    pred_time = perf_counter() - start_time_pred

    average_prediction_time = pred_time / len(X_test) * 1000

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    return train_accuracy, test_accuracy, y_pred, training_time, average_prediction_time

def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, n_estimators=10, data_type="static"):
    if data_type == 'dynamic':
        X_train = flatten_sequences(X_train)
        X_test = flatten_sequences(X_test)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    start_time_train = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - start_time_train

    y_train_pred = model.predict(X_train)

    start_time_pred = perf_counter()
    y_pred = model.predict(X_test)
    pred_time = perf_counter() - start_time_pred

    average_prediction_time = pred_time / len(X_test) * 1000

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    return train_accuracy, test_accuracy, y_pred, training_time, average_prediction_time

def train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel="linear", c=0.1, data_type="static"):
    if data_type == 'dynamic':
        X_train = flatten_sequences(X_train)
        X_test = flatten_sequences(X_test)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(
        kernel=kernel,
        C=c,
        random_state=42
    )
    start_time_train = perf_counter()
    model.fit(X_train_scaled, y_train)
    training_time = perf_counter() - start_time_train

    y_train_pred = model.predict(X_train_scaled)

    start_time_pred = perf_counter()
    y_pred = model.predict(X_test_scaled)
    pred_time = perf_counter() - start_time_pred
    average_prediction_time = pred_time / len(X_test) * 1000

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    return train_accuracy, test_accuracy, y_pred, training_time, average_prediction_time


def train_knn(X_train, y_train, n_neighbors=7):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

