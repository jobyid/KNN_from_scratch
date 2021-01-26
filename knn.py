import numpy as np
import test as tt

class Knn:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        print("this model is to lazy to fit, just go right to prediction")
        return self

    def check_for_missing_values(self, test, train, y_train):
        tc = np.count_nonzero(np.isnan(test))
        trc = np.count_nonzero(np.isnan(train))
        ytc = np.count_nonzero(np.isnan(y_train))
        if tc > 0 or trc > 0 or ytc > 0:
            raise Exception("Data must not contain nan values")


    def find_neighbours(self,x, data, d_class):
        # euc distance for all data points
        # loop all data and find distance from points
        self.neighbours = {}
        for d in data:
            dist = np.linalg.norm(d-x)
            index = np.where(data == d)
            self.neighbours[dist] = d_class[index[0][0]]


    def vote(self):
        keys = np.array(sorted(self.neighbours.keys()))
        # get the k number of lowest value
        cla = keys[:self.k]
        options = []
        for c in cla:
            options.append(self.neighbours[c])
        unique, frequency = np.unique(options, return_counts = True)
        index = np.where(frequency == max(frequency))
        self.classifcation.append(unique[index[0]][0])

    def predict(self,t,X,y):
        self.check_for_missing_values(t,X,y)
        self.classifcation = []
        for x in t:
            self.find_neighbours(x,X,y)
            self.vote()
        print(len(self.classifcation))
        return self.classifcation

# square root of n sometimes a good option for k and should be odd

