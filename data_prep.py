import numpy as np

def process_data_command_line(X_train_csv, y_train_csv, X_test_csv):
    X_train = np.genfromtxt(X_train_csv)
    X_test =  np.genfromtxt(X_test_csv)
    y_train = np.genfromtxt(y_train_csv)
    assert X_train is not None and type(X_train) == np.ndarray, 'X_train should be defined and be a numpy array'
    assert y_train is not None and type(y_train) == np.ndarray, 'y_train should be defined and be a numpy array'
    assert X_test is not None and type(X_test) == np.ndarray, 'y_train should be defined and be a numpy array'
    return X_train, X_test, y_train


