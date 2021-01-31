from sklearn import datasets
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

iris = datasets.load_iris()

iris_X = iris.data #for four features
#iris_X = iris.data[:, :2] #two features

iris_y = iris.target
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-30]]
iris_y_train = iris_y[indices[:-30]]
iris_X_test = iris_X[indices[-30:]]
iris_y_test = iris_y[indices[-30:]]
#Here the validation set has been removed. Merged with the 
#train set that will then make the validation set using k-fold
svc1 = svm.SVC(kernel='linear',C=0.1)
svc2 = svm.SVC(kernel='linear',C=1)
svc3 = svm.SVC(kernel='linear',C=100)
svc4 = svm.SVC(kernel='linear',C=1000)
svc5 = svm.SVC(kernel='linear',C=10000)
svc6 = svm.SVC(kernel='linear',C=10000000)
svc7 = svm.SVC(kernel='linear',C=2)

kf = KFold(n_splits=10)
kf.get_n_splits(iris_X_train)

scores1 = []
scores2 = []
scores3 = []
scores4 = []
scores5 = []
scores6 = []
scores7 = []

for train_index, test_index in kf.split(iris_X_train):
    X_train, X_valid = iris_X_train[train_index], iris_X_train[test_index]
    Y_train, Y_valid = iris_y_train[train_index], iris_y_train[test_index]
    
    svc1.fit(X_train, Y_train)
    svc2.fit(X_train, Y_train)
    svc3.fit(X_train, Y_train)
    svc4.fit(X_train, Y_train)
    svc5.fit(X_train, Y_train)
    svc6.fit(X_train, Y_train)
    svc7.fit(X_train, Y_train)

    
    scores1.append(svc1.score(X_valid, Y_valid))
    scores2.append(svc2.score(X_valid, Y_valid))
    scores3.append(svc3.score(X_valid, Y_valid))
    scores4.append(svc4.score(X_valid, Y_valid))
    scores5.append(svc5.score(X_valid, Y_valid))
    scores6.append(svc6.score(X_valid, Y_valid))
    scores7.append(svc7.score(X_valid, Y_valid))    
    
print("Average Score for C=0.1")
print(np.mean(scores1))
print("Average Score for C=1")
print(np.mean(scores2))
print("Average Score for C=2")
print(np.mean(scores7))
print("Average Score for C=100")
print(np.mean(scores3))
print("Average Score for C=1000")
print(np.mean(scores4))
print("Average Score for C=10000")
print(np.mean(scores5))
print("Average Score for C=10000000")
print(np.mean(scores6))

#After 20 attempts at 10 folds of dataset
#C=1 was chosen. As it gave the highest average cross-validation score.
final_svc = svm.SVC(kernel='linear',C=1)
final_svc.fit(iris_X_train, iris_y_train)
print("Final testing Score for C=2 in linear kernel  BEST RESULT")
print(final_svc.score(iris_X_test,iris_y_test))

