from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def smaple_data():
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X,y_true, test_size=0.1)
    return  X_train,X_test, y_train, y_test

X_train,X_test, y_train, y_test = smaple_data()
np.savetxt("y_test.csv",y_test)
