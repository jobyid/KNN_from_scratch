import knn
import test as tt

knn = knn.Knn(5)
X_train,X_test, y_train, y_test = tt.smaple_data()

print("Length of the test set: ",len(X_test))
preds = knn.predict(X_test,X_train,y_train)
score = 0
for i in range(len(preds)):
    if y_test[i] == preds[i]:
        score += 1
accuracy = score / len(y_test)
print("accuracy of: ",accuracy)
