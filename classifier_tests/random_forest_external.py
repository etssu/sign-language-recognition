from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from experiment_utils import print_model_results, print_test_train_split
from preprocessing import load_static_data_by_person

DATA_FILE = "../data/landmarks.csv"

def train_and_evaluate_random_forest(train_x,test_x,train_y, test_y,n_estimators=10):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(train_x, train_y)

    y_train_pred = model.predict(train_x)
    y_pred = model.predict(test_x)

    train_accuracy = accuracy_score(
        train_y,
        y_train_pred
    )

    test_accuracy = accuracy_score(
        test_y,
        y_pred
    )

    return train_accuracy, test_accuracy, y_pred


# Load data
X_train, X_test, y_train, y_test = load_static_data_by_person(
    DATA_FILE,
    train_person_ids=[2, 3],
    test_person_ids=[1]
)

print_test_train_split(X_train, X_test)


# Random Forest
train_accuracy, test_accuracy, y_pred = train_and_evaluate_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    n_estimators=10
)

print_model_results(
    "RANDOM FOREST",
    train_accuracy,
    test_accuracy,
    y_pred,
    y_test
)