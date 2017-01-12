#########################################################################################
## Run NN on (PCA, ICA, RP, LDA)reduced USPS data:
#########################################################################################
import Create_Dataset
from ULA import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split

USPS = Create_Dataset.create_dataset('/Users/yywxenia/PycharmProjects/MLProj3/USPS_Digit_Data.txt',0,256,256)
X = USPS['data']
y = USPS['target']

# Split iris dataset into 70% train data and 30% test data:
trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.3)
print "Original data size and dimensions:", X.shape
print "Train data size of USPS:", len(trainY)
print "Test data size of USPS:", len(testY)
print '------------------------------------------------------------------------'
components=np.int32(np.linspace(10, 256, 10))
print "Components:", components
print "Original accuracy baseline from NN: "
usps_NN_origin_result = NN(trainX, trainY, testX, testY)
baseline = usps_NN_origin_result[1]
base_line = [baseline] * len(components)
basetime = usps_NN_origin_result[3]* len(components)

print "Baseline test accuracy of USPS NN: ", baseline
print "Base test time of USPS NN: ", basetime
print '------------------------------------------------------------------------'


print "(1) NN accuracy comparison after reducing data by PCA:"
test_pca_acc=[]
test_pca_time=[]

for n_comp in components:
    print "Components number:", n_comp

    X_pca = pca_process(n_comp, X)[0]
    trainX, testX, trainY, testY = train_test_split(X_pca, y, test_size = 0.3)

    usps_NN_pca_result = NN(trainX, trainY, testX, testY)
    NN_pca_acc = usps_NN_pca_result[1]
    test_pca_acc.append(NN_pca_acc)

    NN_pca_time = usps_NN_pca_result[3]
    test_pca_time.append(NN_pca_time)
    print '~~~~~~~~~~~~~~'

print '------------------------------------------------------------------------'
print "(2) NN accuracy comparison after reducing data by ICA:"
test_ica_acc=[]
test_ica_time=[]

for n_comp in components:
    print "Components number:", n_comp

    X_ica = ica_process(n_comp, X)[0]
    trainX, testX, trainY, testY = train_test_split(X_ica, y, test_size = 0.3)

    usps_NN_ica_result = NN(trainX, trainY, testX, testY)
    NN_ica_acc = usps_NN_ica_result[1]
    test_ica_acc.append(NN_ica_acc)

    NN_ica_time = usps_NN_ica_result[3]
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

    X_RPG = rp_process(n_comp, X)[0]
    X_RPS = rp_process(n_comp, X)[1]
    trainX1, testX1, trainY1, testY1 = train_test_split(X_RPG, y, test_size = 0.3)
    trainX2, testX2, trainY2, testY2 = train_test_split(X_RPS, y, test_size = 0.3)

    usps_NN_RPG_result = NN(trainX1, trainY1, testX1, testY1)
    usps_NN_RPS_result = NN(trainX2, trainY2, testX2, testY2)
    NN_RPG_acc = usps_NN_RPG_result[1]
    NN_RPS_acc = usps_NN_RPS_result[1]
    test_RPG_acc.append(NN_RPG_acc)
    test_RPS_acc.append(NN_RPS_acc)

    NN_RPG_t = usps_NN_RPG_result[3]
    NN_RPS_t = usps_NN_RPS_result[3]
    test_RPG_t.append(NN_RPG_t)
    test_RPS_t.append(NN_RPS_t)
    print '~~~~~~~~~~~~~~'

print '------------------------------------------------------------------------'
print "(4) NN accuracy comparison after reducing data by LDA:"
test_lda_acc=[]
test_lda_time=[]

for n_comp in components:
    print "Components number:", n_comp

    X_lda = lda_process(n_comp, X, y)[0]
    trainX, testX, trainY, testY = train_test_split(X_lda, y, test_size = 0.3)

    usps_NN_lda_result = NN(trainX, trainY, testX, testY)
    NN_lda_acc = usps_NN_lda_result[1]
    test_lda_acc.append(NN_lda_acc)

    NN_lda_time = usps_NN_lda_result[3]
    test_lda_time.append(NN_lda_time)
    print '~~~~~~~~~~~~~~'


# Plot the result:
plt.figure()
plt.title("NN accuracy comparison on reduced USPS data")
plt.xlabel("Number of Components")
plt.ylabel("Test Accuracy")
plt.plot(components, base_line, linewidth=2.5, color = "r", linestyle='--', label='Baseline')
plt.plot(components, test_pca_acc, linewidth=1.5, color = "b", label='PCA')
plt.plot(components, test_ica_acc, linewidth=1.5, color = "m", label='ICA')
plt.plot(components, test_RPG_acc, linewidth=1.5, color = "c", label='RP-Gaussian')
plt.plot(components, test_RPS_acc, linewidth=1.5, color = "k", label='RP-Sparse')
plt.plot(components, test_lda_acc, linewidth=1.5, color = "g", label='LDA')
plt.legend(loc='lower right')
plt.show()


# Plot the time result:
plt.figure()
plt.title("NN test time comparison on reduced USPS data")
plt.xlabel("Number of Components")
plt.ylabel("Test Time")
plt.plot(components, basetime, linewidth=2.5, color = "r", linestyle='--', label='Baseline')
plt.plot(components, test_pca_time, linewidth=1.5, color = "b", label='PCA')
plt.plot(components, test_ica_time, linewidth=1.5, color = "m", label='ICA')
plt.plot(components, test_RPG_t, linewidth=1.5, color = "c", label='RP-Gaussian')
plt.plot(components, test_RPS_t, linewidth=1.5, color = "k", label='RP-Sparse')
plt.plot(components, test_lda_time, linewidth=1.5, color = "g", label='LDA')
plt.legend(loc='upper right')
plt.show()

print "===========================THE END================================="
