from ULA import *
import numpy as np
import matplotlib.pyplot as plt
import Create_Dataset
from sklearn.cross_validation import train_test_split

#########################################################################################
### Run NN on clustered (kmeans & em) USPS data:
#########################################################################################

USPS = Create_Dataset.create_dataset('/Users/yywxenia/PycharmProjects/MLProj3/USPS_Digit_Data.txt',0,256,256)
X = USPS['data']
y = USPS['target']
print "USPS data size and dimensions:", X.shape
trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.3)
print "Original data size and dimensions:", X.shape
print "Train data size of USPS:", len(trainY)
print "Test data size of USPS:", len(testY)
print '------------------------------------------------------------------------'
components = np.int32(np.linspace(3, 20, 10))
print "Clusters:", components

print "Original accuracy baseline from NN: "
usps_NN_origin_result = NN(trainX, trainY, testX, testY)
baseline = usps_NN_origin_result[1]
base_line = [baseline] * len(components)
basetime = usps_NN_origin_result[3]*len(components)

print "Baseline test accuracy of USPS NN: ", baseline
print "Base test time of USPS NN: ", basetime
print '------------------------------------------------------------------------'


print "(1) NN accuracy comparison after changing clusters by EM:"
test_em_acc=[]
test_em_time=[]

for n_comp in components:
    print "Clusters number:", n_comp
    X_em = cluster_process('em', n_comp, X, y)[5]
    trainX, testX, trainY, testY = train_test_split(X_em, y, test_size = 0.3)

    usps_em_result = NN(trainX, trainY, testX, testY)
    NN_em_acc = usps_em_result[1]
    test_em_acc.append(NN_em_acc)

    NN_em_time = usps_em_result[3]
    test_em_time.append(NN_em_time)
    print '~~~~~~~~~~~~~~'


print '------------------------------------------------------------------------'
print "(2) NN accuracy comparison after changing clusters by Kmeans:"
test_k_acc=[]
test_k_time=[]

for n_comp in components:
    print "Clusters number:", n_comp
    X_k = cluster_process('kmeans', n_comp, X, y)[6]
    trainX, testX, trainY, testY = train_test_split(X_k, y, test_size = 0.3)

    usps_k_result = NN(trainX, trainY, testX, testY)
    NN_k_acc = usps_k_result[1]
    test_k_acc.append(NN_k_acc)

    NN_k_time = usps_k_result[3]
    test_k_time.append(NN_k_time)
    print '~~~~~~~~~~~~~~'

###### Plot the result:
plt.figure()
plt.title("NN accuracy comparison on 'clustered' USPS data")
plt.xlabel("Number of Clusters")
plt.ylabel("Test Accuracy")
plt.plot(components, base_line, linewidth=2.5, color = "r", linestyle='--', label='Baseline')
plt.plot(components, test_k_acc, linewidth=1.5, color = "b", label='Kmeans')
plt.plot(components, test_em_acc, linewidth=1.5, color = "m", label='EM')
plt.legend(loc='upper left')
plt.show()


# Plot the time result:
plt.figure()
plt.title("NN test time comparison on 'clustered' USPS data")
plt.xlabel("Number of Clusters")
plt.ylabel("Test Time")
plt.plot(components, basetime, linewidth=2.5, color = "r", linestyle='--', label='Baseline')
plt.plot(components, test_k_time, linewidth=1.5, color = "b", label='Kmeans')
plt.plot(components, test_em_time, linewidth=1.5, color = "m", label='EM')
plt.legend(loc='upper left')
plt.show()

print "===========================THE END================================="



