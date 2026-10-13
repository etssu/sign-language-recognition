import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

from preprocessing import load_dynamic_data_by_person
from utils.experiment_utils import print_test_train_split, print_model_results

# Load dataset
DATA_FILE = "../data/landmarks.csv"

def train_and_evaluate_gru(train_x, test_x, train_y, test_y):
    X_train = np.array(train_x)
    X_test = np.array(test_x)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", train_y.shape)
    print("y_test shape:", test_y.shape)

    model = Sequential()
   # model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(GRU(units=50))
    model.add(Dense(units=5, activation='softmax'))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # train the model
    model.fit(X_train, train_y, epochs=10, batch_size=32)

    y_train_pred = np.argmax(model.predict(X_train), axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    train_accuracy = accuracy_score(train_y, y_train_pred)
    test_accuracy = accuracy_score(test_y, y_pred)

    return train_accuracy, test_accuracy, y_pred


X_train, X_test, y_train, y_test, train_seq_len, test_seq_len = load_dynamic_data_by_person(DATA_FILE, [1,2], [3])

selected_gestures = {"Á", "Ä", "Č", "Ď", "É"}

train_mask = np.isin(y_train, list(selected_gestures))
test_mask = np.isin(y_test, list(selected_gestures))

X_train = [seq for seq, keep in zip(X_train, train_mask) if keep]
y_train = y_train[train_mask]

X_test = [seq for seq, keep in zip(X_test, test_mask) if keep]
y_test = y_test[test_mask]


label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print_test_train_split(X_train, y_train)

# GRU
train_accuracy, test_accuracy, y_pred = train_and_evaluate_gru(X_train, X_test, y_train, y_test)

y_test_letters = label_encoder.inverse_transform(y_test)
y_pred_letters = label_encoder.inverse_transform(y_pred)

print_model_results("GRU", train_accuracy, test_accuracy, y_pred_letters, y_test_letters)
