
from experiment_utils import print_model_results, print_test_train_split
from preprocessing import load_dynamic_data_by_person
from model_evaluation import train_and_evaluate_random_forest

DATA_FILE = "../data/landmarks.csv"


# Load data
X_train, X_test, y_train, y_test, train_seq_len, test_seq_len = load_dynamic_data_by_person(DATA_FILE, [3,4], [2])

print_test_train_split(X_train, X_test)


# Random Forest
train_accuracy, test_accuracy, y_pred = train_and_evaluate_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    n_estimators=10,
    data_type="dynamic"
)

print_model_results("RANDOM FOREST",train_accuracy,test_accuracy,y_pred,y_test)