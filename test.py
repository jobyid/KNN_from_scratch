from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np

def smaple_data():
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    return  X, y_true
