import Create_Dataset
from Algorithms import CVtest
from Algorithms import test_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


BANK = Create_Dataset.create_dataset('/Users/yywxenia/Desktop/bank_full.txt',0,20,20)

###(2) Training and Testing data
n_samples = len(BANK['target'])
print "Number of samples: ", n_samples
X0 = BANK['data']
y = BANK['target']
X = Create_Dataset.normalize_featDim(X0)
X = X*1000
X, y = shuffle(X, y, random_state=0)

###(3) Implement different algorithms on the BM dataset
#=============================================================================================
print "==============================================================================="
print " 0st: Using Neural Network:"


print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
# print "[CV_fold = 3]"

meanTrainAcc, meanTestAcc, meanTrainTime, meanTestTime = CVtest(10, X, y, "NN", 50, 10)
NumUnits = range(10, 50+1, 10)

# plt.figure()
plt.plot(NumUnits, meanTrainAcc, "bo-", label="Training accuracy")
plt.plot(NumUnits, meanTestAcc,  "ro-", label="Testing accuracy")
plt.xlabel("Number of nodes in each layer")
plt.ylabel("Training/Testing accuracy")
plt.legend(loc="best")
plt.grid()
plt.title("Neural Network of 3 layers on BANK in 10 fold CV")
plt.savefig("BANK_NN_1HL.png", dpi=500)
# plt.show()


#
NumNN = range(10, 40+1, 10)
print NumNN
cvFoldList = range(2, 10, 2)

Train_Acc_CV = []
Test_Acc_CV  = []
Train_Time_CV= []
Test_Time_CV = []
for cvFold in cvFoldList:
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "[CV_fold = " + str(cvFold) + "]"

    meanTrainAcc, meanTestAcc, meanTrainTime, meanTestTime = CVtest(cvFold, X, y, "NN", 40, 10)
    Train_Acc_CV.append(meanTrainAcc)
    Test_Acc_CV.append(meanTestAcc)
    Train_Time_CV.append(meanTrainTime)
    Test_Time_CV.append(meanTestTime)


Train_Acc_CV = np.array(Train_Acc_CV)
Test_Acc_CV  = np.array(Test_Acc_CV)
Train_Time_CV = np.array(Train_Time_CV)
Test_Time_CV = np.array(Test_Time_CV)

plt.figure(figsize=(13, 8))
# fig = plt.gcf()
# fig.suptitle("NN results (inv.dist. weight + l2 distance) on USPS in variours CV", fontsize=15)
#
ax = plt.subplot(2, 2, 1)
plt.plot(cvFoldList, Train_Acc_CV[:, 0], "bo-", label="#Nodes = 10")
plt.plot(cvFoldList, Train_Acc_CV[:, 1], "ro-", label="#Nodes = 20")
plt.plot(cvFoldList, Train_Acc_CV[:, 2], "go-", label="#Nodes = 30")
plt.plot(cvFoldList, Train_Acc_CV[:, 3], "ko-", label="#Nodes = 40")
plt.ylabel("Training Accuracy")
#ax.set_title("Training Accuracy")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 2)
plt.plot(cvFoldList, Test_Acc_CV[:, 0], "bo-", label="#Nodes = 10")
plt.plot(cvFoldList, Test_Acc_CV[:, 1], "ro-", label="#Nodes = 20")
plt.plot(cvFoldList, Test_Acc_CV[:, 2], "go-", label="#Nodes = 30")
plt.plot(cvFoldList, Test_Acc_CV[:, 3], "ko-", label="#Nodes = 40")
plt.ylabel("Testing Accuracy")
#ax.set_title("Testing Accuracy")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 3)
plt.plot(cvFoldList, Train_Time_CV[:, 0], "bo-", label="#Nodes = 10")
plt.plot(cvFoldList, Train_Time_CV[:, 1], "ro-", label="#Nodes = 20")
plt.plot(cvFoldList, Train_Time_CV[:, 2], "go-", label="#Nodes = 30")
plt.plot(cvFoldList, Train_Time_CV[:, 3], "ko-", label="#Nodes = 40")
plt.ylabel("Training Time")
#ax.set_title("Training Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 4)
plt.plot(cvFoldList, Test_Time_CV[:, 0], "bo-", label="#Nodes = 10")
plt.plot(cvFoldList, Test_Time_CV[:, 1], "ro-", label="#Nodes = 20")
plt.plot(cvFoldList, Test_Time_CV[:, 2], "go-", label="#Nodes = 30")
plt.plot(cvFoldList, Test_Time_CV[:, 3], "ko-", label="#Nodes = 40")
plt.ylabel("Testing Time")
#ax.set_title("Testing Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
plt.savefig("BANK_NN_1HL_All.png", dpi=500)




#===============================================================================================
print "==============================================================================="
print " 1st: Using KNN:"

NumNN = range(1, 8, 2)
print NumNN
cvFoldList = range(2, 15, 1)

Train_Acc_CV = []
Test_Acc_CV  = []
Train_Time_CV= []
Test_Time_CV = []
for cvFold in cvFoldList:
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "[CV_fold = " + str(cvFold) + "]"

    meanTrainAcc, meanTestAcc, meanTrainTime, meanTestTime = CVtest(cvFold, X, y, "Knn", 8, 2)
    Train_Acc_CV.append(meanTrainAcc)
    Test_Acc_CV.append(meanTestAcc)
    Train_Time_CV.append(meanTrainTime)
    Test_Time_CV.append(meanTestTime)


Train_Acc_CV = np.array(Train_Acc_CV)
Test_Acc_CV  = np.array(Test_Acc_CV)
Train_Time_CV = np.array(Train_Time_CV)
Test_Time_CV = np.array(Test_Time_CV)

plt.figure(figsize=(13, 8))
fig = plt.gcf()
fig.suptitle("KNN results (inv.dist. weight + l2 distance) on BM in variours CV", fontsize=15)
#
ax = plt.subplot(2, 2, 1)
plt.plot(cvFoldList, Train_Acc_CV[:, 0], "bo-", label="$k=1$")
plt.plot(cvFoldList, Train_Acc_CV[:, 1], "ro-", label="$k=3$")
plt.plot(cvFoldList, Train_Acc_CV[:, 2], "go-", label="$k=5$")
plt.plot(cvFoldList, Train_Acc_CV[:, 3], "ko-", label="$k=7$")
plt.ylabel("Training Accuracy")
ax.set_title("Training Accuracy")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 2)
plt.plot(cvFoldList, Test_Acc_CV[:, 0], "bo-", label="$k=1$")
plt.plot(cvFoldList, Test_Acc_CV[:, 1], "ro-", label="$k=3$")
plt.plot(cvFoldList, Test_Acc_CV[:, 2], "go-", label="$k=5$")
plt.plot(cvFoldList, Test_Acc_CV[:, 3], "ko-", label="$k=7$")
plt.ylabel("Testing Accuracy")
ax.set_title("Testing Accuracy")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 3)
plt.plot(cvFoldList, Train_Time_CV[:, 0], "bo-", label="$k=1$")
plt.plot(cvFoldList, Train_Time_CV[:, 1], "ro-", label="$k=3$")
plt.plot(cvFoldList, Train_Time_CV[:, 2], "go-", label="$k=5$")
plt.plot(cvFoldList, Train_Time_CV[:, 3], "ko-", label="$k=7$")
plt.ylabel("Training Time")
ax.set_title("Training Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 4)
plt.plot(cvFoldList, Test_Time_CV[:, 0], "bo-", label="$k=1$")
plt.plot(cvFoldList, Test_Time_CV[:, 1], "ro-", label="$k=3$")
plt.plot(cvFoldList, Test_Time_CV[:, 2], "go-", label="$k=5$")
plt.plot(cvFoldList, Test_Time_CV[:, 3], "ko-", label="$k=7$")
plt.ylabel("Testing Time")
ax.set_title("Testing Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
plt.savefig("BankMarketing_KNN_All.png", dpi=500)


# #===============================================================================================
print "==============================================================================="
print " 2nd: Using DT:"

MaxDepth = range(10, 41, 10)
print MaxDepth
cvFoldList = range(2, 15, 1)

Train_Acc_CV = []
Test_Acc_CV  = []
Train_Time_CV= []
Test_Time_CV = []
for cvFold in cvFoldList:
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "[CV_fold = " + str(cvFold) + "]"

    meanTrainAcc, meanTestAcc, meanTrainTime, meanTestTime = CVtest(cvFold, X, y, "DT_pruning", 40, 10)
    Train_Acc_CV.append(meanTrainAcc)
    Test_Acc_CV.append(meanTestAcc)
    Train_Time_CV.append(meanTrainTime)
    Test_Time_CV.append(meanTestTime)


Train_Acc_CV = np.array(Train_Acc_CV)
Test_Acc_CV  = np.array(Test_Acc_CV)
Train_Time_CV = np.array(Train_Time_CV)
Test_Time_CV = np.array(Test_Time_CV)

plt.figure(figsize=(13, 8))
fig = plt.gcf()
fig.suptitle("Decision Tree results (Gini) on BM in variours CV", fontsize=15)
#
ax = plt.subplot(2, 2, 1)
plt.plot(cvFoldList, Train_Acc_CV[:, 0], "bo-", label="Max.Depth=10")
plt.plot(cvFoldList, Train_Acc_CV[:, 1], "ro-", label="Max.Depth=20")
plt.plot(cvFoldList, Train_Acc_CV[:, 2], "go-", label="Max.Depth=30")
plt.plot(cvFoldList, Train_Acc_CV[:, 3], "ko-", label="Max.Depth=40")
plt.ylabel("Training Accuracy")
ax.set_title("Training Accuracy")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 2)
plt.plot(cvFoldList, Test_Acc_CV[:, 0], "bo-", label="Max.Depth=10")
plt.plot(cvFoldList, Test_Acc_CV[:, 1], "ro-", label="Max.Depth=20")
plt.plot(cvFoldList, Test_Acc_CV[:, 2], "go-", label="Max.Depth=30")
plt.plot(cvFoldList, Test_Acc_CV[:, 3], "ko-", label="Max.Depth=40")
plt.ylabel("Testing Accuracy")
ax.set_title("Testing Accuracy")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 3)
plt.plot(cvFoldList, Train_Time_CV[:, 0], "bo-", label="Max.Depth=10")
plt.plot(cvFoldList, Train_Time_CV[:, 1], "ro-", label="Max.Depth=20")
plt.plot(cvFoldList, Train_Time_CV[:, 2], "go-", label="Max.Depth=30")
plt.plot(cvFoldList, Train_Time_CV[:, 3], "ko-", label="Max.Depth=40")
plt.ylabel("Training Time")
ax.set_title("Training Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 4)
plt.plot(cvFoldList, Test_Time_CV[:, 0], "bo-", label="Max.Depth=10")
plt.plot(cvFoldList, Test_Time_CV[:, 1], "ro-", label="Max.Depth=20")
plt.plot(cvFoldList, Test_Time_CV[:, 2], "go-", label="Max.Depth=30")
plt.plot(cvFoldList, Test_Time_CV[:, 3], "ko-", label="Max.Depth=40")
plt.ylabel("Testing Time")
ax.set_title("Testing Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
plt.savefig("BankMarketing_DT_All.png", dpi=500)


# #===============================================================================================
# print "==============================================================================="
# print " 3rd: Using Adaboost DT:"

MaxDepth = range(1, 7, 2)
print MaxDepth
cvFoldList = range(2, 15, 1)

Train_Acc_CV = []
Test_Acc_CV  = []
Train_Time_CV= []
Test_Time_CV = []
for cvFold in cvFoldList:
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "[CV_fold = " + str(cvFold) + "]"

    meanTrainAcc, meanTestAcc, meanTrainTime, meanTestTime = CVtest(cvFold, X, y, "Boost", 7, 2)
    Train_Acc_CV.append(meanTrainAcc)
    Test_Acc_CV.append(meanTestAcc)
    Train_Time_CV.append(meanTrainTime)
    Test_Time_CV.append(meanTestTime)


Train_Acc_CV = np.array(Train_Acc_CV)
Test_Acc_CV  = np.array(Test_Acc_CV)
Train_Time_CV = np.array(Train_Time_CV)
Test_Time_CV = np.array(Test_Time_CV)

plt.figure(figsize=(13, 8))
fig = plt.gcf()
fig.suptitle("Adaboost DT results on BM in variours CV", fontsize=15)
#
ax = plt.subplot(2, 2, 1)
plt.plot(cvFoldList, Train_Acc_CV[:, 0], "bo-", label="Max.Depth=1")
plt.plot(cvFoldList, Train_Acc_CV[:, 1], "ro-", label="Max.Depth=3")
plt.plot(cvFoldList, Train_Acc_CV[:, 2], "go-", label="Max.Depth=5")
plt.plot(cvFoldList, Train_Acc_CV[:, 3], "ko-", label="Max.Depth=7")
plt.ylabel("Training Accuracy")
ax.set_title("Training Accuracy")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 2)
plt.plot(cvFoldList, Test_Acc_CV[:, 0], "bo-", label="Max.Depth=1")
plt.plot(cvFoldList, Test_Acc_CV[:, 1], "ro-", label="Max.Depth=3")
plt.plot(cvFoldList, Test_Acc_CV[:, 2], "go-", label="Max.Depth=5")
plt.plot(cvFoldList, Test_Acc_CV[:, 3], "ko-", label="Max.Depth=7")
plt.ylabel("Testing Accuracy")
ax.set_title("Testing Accuracy")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 3)
plt.plot(cvFoldList, Train_Time_CV[:, 0], "bo-", label="Max.Depth=1")
plt.plot(cvFoldList, Train_Time_CV[:, 1], "ro-", label="Max.Depth=3")
plt.plot(cvFoldList, Train_Time_CV[:, 2], "go-", label="Max.Depth=5")
plt.plot(cvFoldList, Train_Time_CV[:, 3], "ko-", label="Max.Depth=7")
plt.ylabel("Training Time")
ax.set_title("Training Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 4)
plt.plot(cvFoldList, Test_Time_CV[:, 0], "bo-", label="Max.Depth=1")
plt.plot(cvFoldList, Test_Time_CV[:, 1], "ro-", label="Max.Depth=3")
plt.plot(cvFoldList, Test_Time_CV[:, 2], "go-", label="Max.Depth=5")
plt.plot(cvFoldList, Test_Time_CV[:, 3], "ko-", label="Max.Depth=7")
plt.ylabel("Testing Time")
ax.set_title("Testing Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
plt.savefig("BankMarketing_AdaBoostDT_All_2.png", dpi=500)


##===============================================================================================
print "==============================================================================="
print " 4th: Using SVMs:"

SVM_bwList = [0.1]
cvFoldList = range(2, 15, 1)

for SVM_bandwidth in SVM_bwList:
    Train_Acc_CV = []
    Test_Acc_CV  = []
    Train_Time_CV= []
    Test_Time_CV = []
    for cvFold in cvFoldList:
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print "[CV_fold = " + str(cvFold) + ", Gamma = " + str(SVM_bandwidth) + "]"

        meanTrainAcc, meanTestAcc, meanTrainTime, meanTestTime = CVtest(cvFold, X, y, "SVMs", SVM_bandwidth, 0)
        Train_Acc_CV.append(meanTrainAcc)
        Test_Acc_CV.append(meanTestAcc)
        Train_Time_CV.append(meanTrainTime)
        Test_Time_CV.append(meanTestTime)


    Train_Acc_CV = np.array(Train_Acc_CV)
    Test_Acc_CV  = np.array(Test_Acc_CV)
    Train_Time_CV = np.array(Train_Time_CV)
    Test_Time_CV = np.array(Test_Time_CV)

    print Train_Acc_CV

    plt.figure(figsize=(13, 8))
    fig = plt.gcf()
    fig.suptitle("SVM results on BM dataset with hyper-parameter = "+str(SVM_bandwidth), fontsize=15)
    #
    ax = plt.subplot(2, 2, 1)
    plt.plot(cvFoldList, Train_Acc_CV[:, 0], "bo-", label="Linear kernel")
    plt.plot(cvFoldList, Train_Acc_CV[:, 1], "ro-", label="Polynomial kernel")
    plt.plot(cvFoldList, Train_Acc_CV[:, 2], "go-", label="RBF kernel")
    plt.plot(cvFoldList, Train_Acc_CV[:, 3], "ko-", label="Sigmoid kernel")
    plt.ylabel("Training Accuracy")
    ax.set_title("Training Accuracy")
    plt.grid()
    plt.legend(loc="best",prop={'size':10})
    plt.xlabel("Number of folds in Cross-Validation")
    #
    ax = plt.subplot(2, 2, 2)
    plt.plot(cvFoldList, Test_Acc_CV[:, 0], "bo-", label="Linear kernel")
    plt.plot(cvFoldList, Test_Acc_CV[:, 1], "ro-", label="Polynomial kernel")
    plt.plot(cvFoldList, Test_Acc_CV[:, 2], "go-", label="RBF kernel")
    plt.plot(cvFoldList, Test_Acc_CV[:, 3], "ko-", label="Sigmoid kernel")
    plt.ylabel("Testing Accuracy")
    ax.set_title("Testing Accuracy")
    plt.grid()
    plt.legend(loc="best",prop={'size':10})
    plt.xlabel("Number of folds in Cross-Validation")
    #
    ax = plt.subplot(2, 2, 3)
    plt.plot(cvFoldList, Train_Time_CV[:, 0], "bo-", label="Linear kernel")
    plt.plot(cvFoldList, Train_Time_CV[:, 1], "ro-", label="Polynomial kernel")
    plt.plot(cvFoldList, Train_Time_CV[:, 2], "go-", label="RBF kernel")
    plt.plot(cvFoldList, Train_Time_CV[:, 3], "ko-", label="Sigmoid kernel")
    plt.ylabel("Training Time")
    ax.set_title("Training Time")
    plt.grid()
    plt.legend(loc="best",prop={'size':10})
    plt.xlabel("Number of folds in Cross-Validation")
    #
    ax = plt.subplot(2, 2, 4)
    plt.plot(cvFoldList, Test_Time_CV[:, 0], "bo-", label="Linear kernel")
    plt.plot(cvFoldList, Test_Time_CV[:, 1], "ro-", label="Polynomial kernel")
    plt.plot(cvFoldList, Test_Time_CV[:, 2], "go-", label="RBF kernel")
    plt.plot(cvFoldList, Test_Time_CV[:, 3], "ko-", label="Sigmoid kernel")
    plt.ylabel("Testing Time")
    ax.set_title("Testing Time")
    plt.grid()
    plt.legend(loc="best",prop={'size':10})
    plt.xlabel("Number of folds in Cross-Validation")
    #
    plt.savefig("BankMarketing_SVM_hyperparm"+str(SVM_bandwidth)+".png", dpi=500)


print "***FINISHED***"





