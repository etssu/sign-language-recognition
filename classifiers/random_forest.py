from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from experiment_utils import print_test_train_split, print_model_results
from preprocessing import load_static_data

# 1. Load dataset
DATA_FILE = "../data/landmarks.csv"

def train_and_evaluate_random_forest(train_x,test_x,train_y, test_y,n_estimators=10):
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    rf_classifier.fit(train_x, train_y)

    y_train_pred = rf_classifier.predict(train_x)
    y_pred = rf_classifier.predict(test_x)

    train_accuracy = accuracy_score(
        train_y,
        y_train_pred
    )

    test_accuracy = accuracy_score(
        test_y,
        y_pred
    )

    return train_accuracy, test_accuracy, y_pred

X, y = load_static_data(DATA_FILE)

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print_test_train_split(X_train, X_test)

# 3. Train and evaluate
train_accuracy, test_accuracy, y_pred = train_and_evaluate_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    n_estimators=10
)

print_model_results("Random Forest",
                    train_accuracy,
                    test_accuracy,
                    y_pred,
                    y_test)
