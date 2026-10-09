from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocessing import flatten_sequences


def train_and_evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors=10, data_type="static"):
    if data_type == 'dynamic':
        X_train = flatten_sequences(X_train)
        X_test = flatten_sequences(X_test)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    return train_accuracy, test_accuracy, y_pred

def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, n_estimators=10, data_type="static"):
    if data_type == 'dynamic':
        X_train = flatten_sequences(X_train)
        X_test = flatten_sequences(X_test)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    return train_accuracy, test_accuracy, y_pred

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

    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    return train_accuracy, test_accuracy, y_pred


def train_knn(X_train, y_train, n_neighbors=7):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

