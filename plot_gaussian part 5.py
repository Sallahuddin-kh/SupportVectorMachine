from sklearn import datasets
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
iris = datasets.load_iris()
iris_y = iris.target
iris_X = iris.data[:, :2]
#The permutation of the indexes has been removed to make the result easier to observe
#iris_X_train = iris_X[:-40]
#iris_y_train = iris_y[:-40]
#iris_X_valid = iris_X[-40:-20]
#iris_y_valid = iris_y[-40:-20]
#iris_X_test = iris_X[-20:]
#iris_y_test = iris_y[-20:]
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-40]]
iris_y_train = iris_y[indices[:-40]]
iris_X_valid = iris_X[indices[-40:-20]]
iris_y_valid = iris_y[indices[-40:-20]]
iris_X_test = iris_X[indices[-20:]]
iris_y_test = iris_y[indices[-20:]]


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    return xx, yy

g=1.75#Best value
c=5   #Best value

print("gamma="+str(g)+" C="+str(c)+" BEST VALUES")
svc = svm.SVC(kernel='rbf', gamma=g, C=c)
pred = svc.fit(iris_X_train, iris_y_train)
U, V = iris_X_train[:, 0], iris_X_train[:, 1]
xx, yy = make_meshgrid(U, V)
figsize = 10
fig = plt.figure(figsize=(figsize,figsize))
ax = plt.subplot(111)
plot_contours(ax, svc, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(U, V, c=iris_y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
plt.show()
print("SCORE ON ABOVE GIVEN VALUES")
print(svc.score(iris_X_test,iris_y_test))



