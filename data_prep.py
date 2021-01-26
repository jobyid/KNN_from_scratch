import numpy as np

def process_data_command_line(X_train_csv, y_train_csv, X_test_csv):
    X_train = np.genfromtxt(X_train_csv)
    X_test =  np.genfromtxt(X_test_csv)
    y_train = np.genfromtxt(y_train_csv)

    return X_train, X_test, y_train


