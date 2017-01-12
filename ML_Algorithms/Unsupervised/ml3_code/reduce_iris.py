from ULA import *
import numpy as np
import matplotlib.pyplot as plt
import Create_Dataset
from sklearn.cross_validation import train_test_split

iris = Create_Dataset.create_dataset('/Users/yywxenia/PycharmProjects/MLProj3/iris.data.txt',0,4,4)
X = iris['data']
y = iris['target']

#########################################################################################
### Part1: Implement PCA, ICA, RP, LDA algorithms on iris and USPS datasets:
#########################################################################################

### (1) PCA:
print 'Apply PCA reduction on iris:'
print '\n'
pca_process(4, X)      # 92.5% of the data can be described by the first components.
print "Data shape:", X.shape

print "------------------------------------------------------------"
## Select components by changing total explained variance ratio:
score_list=[]
time_list=[]
num_c=[]

pent=[0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
for c_percent in pent:
    result = pca_process(c_percent, X)
    num_c.append(result[4])
    score_list.append(result[2])
    time_list.append(result[3])

print "Scores: ", score_list
print "Train time: ", time_list
print "Number of components selected: ", num_c


plt.subplot(311)
plt.plot(pent, num_c,"r.-", label='Selected components')
plt.legend(loc='lower right')
plt.ylabel('Components')
plt.xlabel('Pre-set total variance explained ratio')
plt.title("PCA: select components for iris data")

plt.subplot(312)
plt.plot(num_c, score_list, 'b.-', label='Real total explained variance ratio')
plt.ylabel('Results')
plt.xlabel('Number of components')
plt.legend(loc='lower right')

plt.subplot(313)
plt.plot(num_c, time_list, 'b.-', label='Training time')
plt.ylabel('Train time')
plt.xlabel('Number of components')
plt.legend(loc='lower right')
plt.show()

print "===========================Finished PCA==============================="


print 'Apply ICA reduction on iris:'
print '\n'
ica_process(4, X)      # 92.5% of the data can be described by the first components.
print "Data shape:", X.shape

print "------------------------------------------------------------"
## Select components by changing total explained variance ratio:

time_list=[]

num = [1,2,3,4]
for c in num:
    result = ica_process(c, X)
    time_list.append(result[1])

print "Train time: ", time_list
print "Number of components selected: ", num

plt.plot(num,time_list, "r.-", label='Selected components')
plt.legend(loc='lower right')
plt.ylabel('Components')
plt.xlabel('Train time')
plt.title("ICA: iris data")
plt.show()


### Plot the 2-component's iris data (PCA & ICA):
iris_2 = pca_process(2, X)[0]
f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(111)
ax.scatter(iris_2[:,0], iris_2[:, 1], c=y, alpha=0.5)
ax.set_title("PCA: 2-components result: iris data")

iris_2 = ica_process(2, X)[0]
f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(111)
ax.scatter(iris_2[:,0], iris_2[:, 1], c=y, alpha=0.5)
ax.set_title("ICA: 2-components result: iris data")

print "===========================Finished ICA==============================="

components = np.int32(np.linspace(2, 4, 3))
G_time=[]
S_time=[]
for n_comp in components:
    # create the random projection
    a = rp_process(n_comp, X)
    G_time.append(a[2])
    S_time.append(a[3])
    print '~~~~~~~~~~~~~~~~~~~~~~~'

plt.plot(components, G_time,"c.-", components, S_time,'m.-')
plt.legend(['Gauissian RP', 'Sparse RP'], loc='lower right')
plt.ylabel('Time')
plt.xlabel('Components')
plt.title("Time change by increasing number of components")
plt.show()

print "=====================Finished Random Projection======================="

num = [1,2,3,4]
time_list=[]
for c in num:
    result = lda_process(c, X, y)
    time_list.append(result[1])

print "Train time: ", time_list
print "Number of components selected: ", num

plt.plot(num, time_list, "m.-", label='Time')
plt.legend(loc='lower right')
plt.xlabel('Components')
plt.ylabel('Train time')
plt.title("LDA: iris data")
plt.show()

### Plot the 2-component's iris data (PCA & ICA):
iris_2 = pca_process(2, X)[0]
f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(111)
ax.scatter(iris_2[:,0], iris_2[:, 1], c=y, alpha=0.5)
ax.set_title("PCA: 2-components result: iris data")

iris_2 = lda_process(2, X, y)[0]
f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(111)
ax.scatter(iris_2[:,0], iris_2[:, 1], c=y, alpha=0.5)
ax.set_title("LDA: 2-components result: iris data")

print "===========================Finished LDA==============================="

#########################################################################################
### Part2: Run NN on reduced USPS data: (PCA, ICA, RP, LDA)
#########################################################################################

trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.3)
print "Original data size and dimensions:", X.shape
print "Train data size of iris:", len(trainY)
print "Test data size of iris:", len(testY)
print '------------------------------------------------------------------------'
print "Original accuracy baseline from NN: "
iris_NN_origin_result = NN(trainX, trainY, testX, testY)
baseline = iris_NN_origin_result[1]
base_line = [baseline] * 4
basetime = iris_NN_origin_result[3] * 4
print "Baseline test accuracy of iris NN: ", baseline
print "Base test time of iris NN: ", basetime
print '------------------------------------------------------------------------'
components=[1,2,3,4]

print "(1) NN accuracy comparison after reducing data by PCA:"
test_pca_acc=[]
test_pca_time=[]

for n_comp in components:
    print "Components number:", n_comp
    # Get new reduced iris data from PCA:
    X_pca = pca_process(n_comp, X)[0]
    trainX, testX, trainY, testY = train_test_split(X_pca, y, test_size = 0.3)
    # Apply in NN model to compute the accuracy:
    iris_NN_pca_result = NN(trainX, trainY, testX, testY)
    NN_pca_acc = iris_NN_pca_result[1]
    test_pca_acc.append(NN_pca_acc)

    NN_pca_time = iris_NN_pca_result[3]
    test_pca_time.append(NN_pca_time)
    print '~~~~~~~~~~~~~~'

print '------------------------------------------------------------------------'
print "(2) NN accuracy comparison after reducing data by ICA:"
test_ica_acc=[]
test_ica_time=[]

for n_comp in components:
    print "Components number:", n_comp
    # Get new reduced iris data from PCA:
    X_ica = ica_process(n_comp, X)[0]
    trainX, testX, trainY, testY = train_test_split(X_ica, y, test_size = 0.3)
    # Apply in NN model to compute the accuracy:
    iris_NN_ica_result = NN(trainX, trainY, testX, testY)
    NN_ica_acc = iris_NN_ica_result[1]
    test_ica_acc.append(NN_ica_acc)

    NN_ica_time = iris_NN_ica_result[3]
    test_ica_time.append(NN_ica_time)
    print '~~~~~~~~~~~~~~'

print '------------------------------------------------------------------------'
print "(3) NN accuracy comparison after reducing data by RP:"
test_RPG_acc=[]
test_RPS_acc=[]
test_RPG_t=[]
test_RPS_t=[]

for n_comp in components:
    print "Components number:", n_comp
    # Get new reduced iris data from PCA:
    X_RPG = rp_process(n_comp, X)[0]
    X_RPS = rp_process(n_comp, X)[1]
    trainX1, testX1, trainY1, testY1 = train_test_split(X_RPG, y, test_size = 0.3)
    trainX2, testX2, trainY2, testY2 = train_test_split(X_RPS, y, test_size = 0.3)
    # Apply in NN model to compute the accuracy:
    iris_NN_RPG_result = NN(trainX1, trainY1, testX1, testY1)
    iris_NN_RPS_result = NN(trainX2, trainY2, testX2, testY2)
    NN_RPG_acc = iris_NN_RPG_result[1]
    NN_RPS_acc = iris_NN_RPS_result[1]
    test_RPG_acc.append(NN_RPG_acc)
    test_RPS_acc.append(NN_RPS_acc)

    NN_RPG_t = iris_NN_RPG_result[3]
    NN_RPS_t = iris_NN_RPS_result[3]
    test_RPG_t.append(NN_RPG_t)
    test_RPS_t.append(NN_RPS_t)
    print '~~~~~~~~~~~~~~'

print '------------------------------------------------------------------------'
print "(4) NN accuracy comparison after reducing data by LDA:"
test_lda_acc=[]
test_lda_time=[]

for n_comp in components:
    print "Components number:", n_comp
    # Get new reduced iris data from PCA:
    X_lda = lda_process(n_comp, X, y)[0]
    trainX, testX, trainY, testY = train_test_split(X_lda, y, test_size = 0.3)
    # Apply in NN model to compute the accuracy:
    iris_NN_lda_result = NN(trainX, trainY, testX, testY)
    NN_lda_acc = iris_NN_lda_result[1]
    test_lda_acc.append(NN_lda_acc)

    NN_lda_time = iris_NN_lda_result[3]
    test_lda_time.append(NN_lda_time)
    print '~~~~~~~~~~~~~~'


# Plot the result:
plt.figure()
plt.title("NN accuracy comparison on reduced iris data")
plt.xlabel("Number of Components")
plt.ylabel("Test Accuracy")
plt.plot(components, base_line, linewidth=2.5, color = "r", linestyle='--', label='Baseline Accuracy')
plt.plot(components, test_pca_acc, linewidth=1.5, color = "b", label='PCA test accuracy')
plt.plot(components, test_ica_acc, linewidth=1.5, color = "m", label='ICA test accuracy')
plt.plot(components, test_RPG_acc, linewidth=1.5, color = "c", label='RP-Gaussian test accuracy')
plt.plot(components, test_RPS_acc, linewidth=1.5, color = "k", label='RP-Sparse test accuracy')
plt.plot(components, test_lda_acc, linewidth=1.5, color = "g", label='LDA test accuracy')
plt.legend(loc='lower right')
plt.show()


# Plot the time result:
plt.figure()
plt.title("NN test time comparison on reduced iris data")
plt.xlabel("Number of Components")
plt.ylabel("Test Time")
plt.plot(components, basetime, linewidth=2.5, color = "r", linestyle='--', label='Baseline test time')
plt.plot(components, test_pca_time, linewidth=1.5, color = "b", label='PCA test time')
plt.plot(components, test_ica_time, linewidth=1.5, color = "m", label='ICA test time')
plt.plot(components, test_RPG_t, linewidth=1.5, color = "c", label='RP-Gaussian test time')
plt.plot(components, test_RPS_t, linewidth=1.5, color = "k", label='RP-Sparse test time')
plt.plot(components, test_lda_time, linewidth=1.5, color = "g", label='LDA test time')
plt.legend(loc='upper right')
plt.show()

print "===========================THE END================================="

