import numpy as np
import test as tt

class Knn:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        return "this model is to lazy to fit, just go right to prediction"

    def check_for_missing_values(self, test, train, y_train):
        pass

    def find_neighbours(self,x, data, d_class):
        # euc distance for all data points
        # loop all data and find distance from points
        self.neighbours = {}
        for d in data:
            dist = np.linalg.norm(d-x)
            print(dist)
            index = np.where(data == d)
            #print(index, index[0][0])
            print("class should be: ", d_class[index[0][0]])
            self.neighbours[dist] = d_class[index[0][0]]
        #print(self.neighbours)

    def vote(self):
        keys = np.array(sorted(self.neighbours.keys()))
        # get the k number of lowest value
        cla = keys[:self.k]
        #print(cla)
        options = []
        for c in cla:
            options.append(self.neighbours[c])
        unique, frequency = np.unique(options, return_counts = True)
        index = np.where(frequency == max(frequency))
        #print(unique, frequency, index[0])#frequency.index(max(frequency))
        print("Unique is ", unique)
        print("The Class is ", unique[index[0]])
        self.classifcation.append(unique[index[0]][0])
        #print("classifaction is ",self.classifcation)

    def predict(self,t,X,y):
        self.classifcation = []
        for x in t:
            self.find_neighbours(x,X,y)
            self.vote()
        print(len(self.classifcation))
        return self.classifcation

# square root of n sometimes a good option for k and should be odd

