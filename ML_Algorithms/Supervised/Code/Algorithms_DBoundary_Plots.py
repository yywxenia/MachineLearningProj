print(__doc__)
from sklearn import neighbors
import Create_Dataset
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
print(__doc__)
from sklearn import svm
print(__doc__)
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from pylab import *
import numpy as np
import os
import pydot

h = .02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


### 1. Visualization of iris dataset
###=================================================================================================================
iris=Create_Dataset.create_dataset('/Users/yywxenia/Desktop/iris.data.txt',0,4,4)
X=iris['data']
y=iris['target']


def scatter_plot(dim1, dim2):
    for t, marker, color in zip(xrange(3), "^so", "cyr"):
        areas = np.pi * 3 ** 2
        plt.scatter(X[y == t, dim1], X[y == t, dim2], marker=marker, c=color, s=areas, alpha=0.5)
        #y=0,1,2, is in order to seperate different flowers; dim is used to seperate different flower features.
    dim_meaning = {0:'setal length',1:'setal width',2:'petal length',3:'petal width'}
    plt.xlabel(dim_meaning.get(dim1))
    plt.ylabel(dim_meaning.get(dim2))

fig = plt.figure()
fig.subplots_adjust(hspace=0.4,wspace=0.3)

plt.subplot(321)
scatter_plot(0,1) # first and second features' scatter
plt.subplot(322)
scatter_plot(0,2) # first and third features' scatter
plt.subplot(323)
scatter_plot(0,3) # first and fourth features' scatter
plt.subplot(324)
scatter_plot(1,2) # second and third features' scatter
plt.subplot(325)
scatter_plot(1,3) # second and fourth features' scatter
plt.subplot(326) # third and fourth features' scatter
scatter_plot(2,3)
plt.show()

fig = plt.figure(1, figsize = (8, 6))
ax = Axes3D(fig, elev = -150, azim = 110)
X_reduced = PCA(n_components = 3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c =y, cmap = plt.cm.Paired)
ax.set_title("Visualization of iris dataset (after PCA transformation)")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])



### 2. Visualization of algorithms' decision boundary
###================================================================================================================
iris=Create_Dataset.create_dataset('/Users/yywxenia/Desktop/iris.data.txt',0,4,4)
n_samples = len(iris['target'])
idx = Create_Dataset.Shuffle_Data(n_samples)
Shuffle_X_Train = iris['data'][idx]
Shuffle_y_Train = iris['target'][idx]
X = Create_Dataset.PCA_result(2, Shuffle_X_Train)
 # Use PCA to decrease dimension in order to visualize decision-boundary
y = Shuffle_y_Train


###(1) Plot AdaBoost Boundary
### ---------------------------------------------------------------------------------------
clf_1 = DecisionTreeClassifier(max_depth = 4)
clf_2 = AdaBoostClassifier(clf_1, n_estimators = 30)
clf = clf_2.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Boosting: 3-Class classification on iris data" )

clf = neighbors.KNeighborsClassifier(5, weights='distance')
clf.fit(X, y)


###(2) Plot K-nn Boundary
### ----------------------------------------------------------------------------------------
clf = neighbors.KNeighborsClassifier(5, weights='distance')
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("K-nn: 3-Class classification on iris data" )


###(3) Plot Decision Tree Boundary
### ----------------------------------------------------------------------------------------
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf = clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Decision Tree: 3-Class classification on iris data" )
plt.show()


## Print decision tree graph in .pdf
from sklearn.externals.six import StringIO
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6)
clf = clf.fit(Shuffle_X_Train, Shuffle_y_Train)
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
os.unlink('iris.dot')
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris_Depth_test.pdf")


###(4) Plot SVMs Boundary
### ----------------------------------------------------------------------------------------
plt.subplot(311)
clf = svm.SVC(kernel='linear', gamma=2)
clf.fit(X, y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.title('SVMs classification using linear, poly, and rbf kernels')


plt.subplot(312)
clf = svm.SVC(kernel='poly', gamma=2)
clf.fit(X, y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)

plt.subplot(313)
clf = svm.SVC(kernel='rbf', gamma=2)
clf.fit(X, y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.legend()

plt.show()