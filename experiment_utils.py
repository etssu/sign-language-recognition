import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def print_dynamic_dataset_info(X, y, seq_lengths):
    print("\n=========================")
    print("DYNAMIC DATA")
    print("=========================")

    print("Number of sequences:", len(X))
    print("X shape:", X.shape)
    print("Classes:", np.unique(y))
    print("Number of classes:", len(np.unique(y)))

    print("\nSequence lengths:")
    print("Min:", seq_lengths.min())
    print("Max:", seq_lengths.max())
    print("Average:", seq_lengths.mean())

    print("\nSamples per gesture:")
    for gesture in np.unique(y):
        print(gesture, ":", np.sum(y == gesture))


def print_test_train_split(X_train, X_test):
    print("\n=========================")
    print("TRAIN / TEST SPLIT")
    print("=========================")

    print("Training sequences:", len(X_train))
    print("Testing sequences:", len(X_test))

def print_model_results(model, train_accuracy, test_accuracy, y_pred, y_test):
    print("\n=========================")
    print(f"{model} RESULTS")
    print("=========================")

    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test)) # unique labels
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap="Blues") # change color
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.suptitle('Confusion matrix', fontsize=20)
    plt.show()
    # print("\nConfusion matrix:")
    # print(cm)
