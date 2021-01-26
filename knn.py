import numpy as np
import test as tt

class Knn:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        return "this model is to lazy to fit, just go right to prediction"


    def find_neighbours(self,x, data, d_class):
        # euc distance for all data points
        # loop all data and find distance from points
        self.neighbours = {}
        for d in data:
            dist = np.linalg.norm(d-x)
            print(dist)
            index = np.where(data == d)
            print(index, index[0][0])
            self.neighbours[dist] = d_class[index[0][0]]
        #print(self.neighbours)

    def vote(self):
        keys = np.array(sorted(self.neighbours.keys()))
        #keys = keys.sort()
        print(keys)
        # get the k number of lowest value
        cla = keys[:self.k]
        print(cla)
        options = []
        for c in cla:
            options.append(self.neighbours[c])
        unique, frequency = np.unique(options, return_counts = True)
        index = np.where(frequency == max(frequency))
        print(unique, frequency, index[0])#frequency.index(max(frequency))
        self.classifcation = unique[index[0]]
        print(self.classifcation)

    def predict(self,t,X,y):
        self.find_neighbours(t,X,y)
        self.vote()
        return self.classifcation

# square root of n sometimes a good option for k and should be odd
knn = Knn(8)
X, y_true = tt.smaple_data()
X_train = X[:-1,:]
y_true_train = y_true[:-1]
x_pred = X[-1:,:]
print(x_pred)
knn.predict(x_pred,X_train,y_true_train)
