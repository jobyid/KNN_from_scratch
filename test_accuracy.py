import knn
import test as tt

knn = knn.Knn(5)
X_train,X_test, y_train, y_test = tt.smaple_data()

print("Length of the test set: ",len(X_test))
print(knn.predict(X_test,X_train,y_train))
