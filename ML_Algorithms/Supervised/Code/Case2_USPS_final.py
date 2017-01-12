import Create_Dataset
from Algorithms import CVtest
from Algorithms import test_data
# import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from sknn.mlp import Classifier, Layer


USPS = Create_Dataset.create_dataset('/Users/yywxenia/Desktop/USPS_Digit_Data.txt',0,256,256)
###(2) Training and Testing data
n_samples = len(USPS['target'])
print "Number of samples: ", n_samples
X = USPS['data']
y = USPS['target']

# (3) Implement different algorithms on the USPS dataset

#===============================================================================================
print "==============================================================================="
print " 0st: Using Neural Network:"


print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
# print "[CV_fold = 10]"

meanTrainAcc, meanTestAcc, meanTrainTime, meanTestTime = CVtest(10, X, y, "NN", 80, 10)
NumUnits = range(10, 80+1, 10)

# plt.figure()
plt.plot(NumUnits, meanTrainAcc, "bo-", label="Training accuracy")
plt.plot(NumUnits, meanTestAcc,  "ro-", label="Testing accuracy")
plt.xlabel("Number of nodes in each layer")
plt.ylabel("Training/Testing accuracy")
plt.legend(loc="best")
plt.grid()
plt.title("Neural Network: 4-layer (2 hidden) on USPS in 10 fold CV")
plt.savefig("USPS_NN_2HL.png", dpi=500)
# plt.show()

#
NumNN = range(15, 60+1, 15)
print NumNN
cvFoldList = range(2, 10, 2)

Train_Acc_CV = []
Test_Acc_CV  = []
Train_Time_CV= []
Test_Time_CV = []
for cvFold in cvFoldList:
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "[CV_fold = " + str(cvFold) + "]"

    meanTrainAcc, meanTestAcc, meanTrainTime, meanTestTime = CVtest(cvFold, X, y, "NN", 60, 15)
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
plt.plot(cvFoldList, Train_Acc_CV[:, 0], "bo-", label="#Nodes = 15")
plt.plot(cvFoldList, Train_Acc_CV[:, 1], "ro-", label="#Nodes = 30")
plt.plot(cvFoldList, Train_Acc_CV[:, 2], "go-", label="#Nodes = 45")
plt.plot(cvFoldList, Train_Acc_CV[:, 3], "ko-", label="#Nodes = 60")
plt.ylabel("Training Accuracy")
#ax.set_title("Training Accuracy")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 2)
plt.plot(cvFoldList, Test_Acc_CV[:, 0], "bo-", label="#Nodes = 15")
plt.plot(cvFoldList, Test_Acc_CV[:, 1], "ro-", label="#Nodes = 30")
plt.plot(cvFoldList, Test_Acc_CV[:, 2], "go-", label="#Nodes = 45")
plt.plot(cvFoldList, Test_Acc_CV[:, 3], "ko-", label="#Nodes = 60")
plt.ylabel("Testing Accuracy")
#ax.set_title("Testing Accuracy")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 3)
plt.plot(cvFoldList, Train_Time_CV[:, 0], "bo-", label="#Nodes = 15")
plt.plot(cvFoldList, Train_Time_CV[:, 1], "ro-", label="#Nodes = 30")
plt.plot(cvFoldList, Train_Time_CV[:, 2], "go-", label="#Nodes = 45")
plt.plot(cvFoldList, Train_Time_CV[:, 3], "ko-", label="#Nodes = 60")
plt.ylabel("Training Time")
#ax.set_title("Training Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
x = plt.subplot(2, 2, 4)
plt.plot(cvFoldList, Test_Time_CV[:, 0], "bo-", label="#Nodes = 15")
plt.plot(cvFoldList, Test_Time_CV[:, 1], "ro-", label="#Nodes = 30")
plt.plot(cvFoldList, Test_Time_CV[:, 2], "go-", label="#Nodes = 45")
plt.plot(cvFoldList, Test_Time_CV[:, 3], "ko-", label="#Nodes = 60")
plt.ylabel("Testing Time")
#ax.set_title("Testing Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
plt.savefig("USPS_NN_2HL_All_B.png", dpi=500)
#



print "==============================================================================="
print " 1st: Using KNN:"


print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "[CV_fold = 10]"

meanTrainAcc, meanTestAcc, meanTrainTime, meanTestTime = CVtest(10, X, y, "Knn", 70, 5)
NumNN = range(1, 71, 5)

plt.figure()
plt.title("KNN results (uniform weight + l2 distance) on USPS in 10 fold CV")
plt.plot(NumNN, meanTrainAcc, "bo-", label="Training accuracy")
plt.plot(NumNN, meanTestAcc,  "ro-", label="Testing accuracy")
plt.xlabel("\#Neighbors ($k$)")
plt.ylabel("Training/Testing accuracy")
plt.legend()
plt.grid()
plt.savefig("USPS_KNN_SweepK_UniformWeight_L2.png", dpi=500)


NumNN = range(1, 8, 2)
print NumNN
cvFoldList = range(2, 10, 2)

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
fig.suptitle("KNN results (inv.dist. weight + l2 distance) on USPS in variours CV", fontsize=15)
#
ax = plt.subplot(2, 2, 1)
plt.plot(cvFoldList, Train_Acc_CV[:, 0], "bo-", label="$k=1$")
plt.plot(cvFoldList, Train_Acc_CV[:, 1], "ro-", label="$k=3$")
plt.plot(cvFoldList, Train_Acc_CV[:, 2], "go-", label="$k=5$")
plt.plot(cvFoldList, Train_Acc_CV[:, 3], "ko-", label="$k=7$")
plt.ylabel("Training Accuracy")
#ax.set_title("Training Accuracy")
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
#ax.set_title("Testing Accuracy")
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
#ax.set_title("Training Time")
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
#ax.set_title("Testing Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
plt.savefig("USPS_KNN_All.png", dpi=500)


#===============================================================================================
print "==============================================================================="
print " 2nd: Using DT:"

print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "[CV_fold = 10]"

meanTrainAcc, meanTestAcc, meanTrainTime, meanTestTime = CVtest(10, X, y, "DT_pruning", 100, 10)
MaxDepth = range(10, 101, 10)

plt.figure()
plt.title("Decision Trees (Gini) results on USPS in 10 fold CV")
plt.plot(MaxDepth, meanTrainAcc, "bo-", label="Training accuracy")
plt.plot(MaxDepth, meanTestAcc,  "ro-", label="Testing accuracy")
plt.xlabel("Max depth of decision tree")
plt.ylabel("Training/Testing accuracy")
plt.legend()
plt.grid()
plt.savefig("USPS_DT_SweepD_Gini_"+str(np.mean(meanTestAcc))+".png", dpi=500)



MaxDepth = range(10, 41, 10)
print MaxDepth
cvFoldList = range(2, 10, 2)

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
fig.suptitle("Decision Tree results (Gini) on USPS in variours CV", fontsize=15)
#
ax = plt.subplot(2, 2, 1)
plt.plot(cvFoldList, Train_Acc_CV[:, 0], "bo-", label="Max.Depth=10")
plt.plot(cvFoldList, Train_Acc_CV[:, 1], "ro-", label="Max.Depth=20")
plt.plot(cvFoldList, Train_Acc_CV[:, 2], "go-", label="Max.Depth=30")
plt.plot(cvFoldList, Train_Acc_CV[:, 3], "ko-", label="Max.Depth=40")
plt.ylabel("Training Accuracy")
#ax.set_title("Training Accuracy")
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
#ax.set_title("Testing Accuracy")
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
#ax.set_title("Training Time")
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
#ax.set_title("Testing Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
plt.savefig("USPS_DT_All.png", dpi=500)



#
# #===============================================================================================
# print "==============================================================================="
# print " 3rd: Using Adaboost DT:"

MaxDepth = range(1, 7, 2)
print MaxDepth
cvFoldList = range(2, 10, 2)

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
fig.suptitle("Adaboost DT results on USPS in variours CV", fontsize=15)
#
ax = plt.subplot(2, 2, 1)
plt.plot(cvFoldList, Train_Acc_CV[:, 0], "bo-", label="Max.Depth=1")
plt.plot(cvFoldList, Train_Acc_CV[:, 1], "ro-", label="Max.Depth=3")
plt.plot(cvFoldList, Train_Acc_CV[:, 2], "go-", label="Max.Depth=5")
plt.plot(cvFoldList, Train_Acc_CV[:, 3], "ko-", label="Max.Depth=7")
plt.ylabel("Training Accuracy")
#ax.set_title("Training Accuracy")
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
#ax.set_title("Testing Accuracy")
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
#ax.set_title("Training Time")
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
#ax.set_title("Testing Time")
plt.grid()
plt.legend(loc="best",prop={'size':10})
plt.xlabel("Number of folds in Cross-Validation")
#
plt.savefig("USPS_AdaBoostDT_All_2.png", dpi=500)



# ===============================================================================================
print "==============================================================================="
print " 4th: Using SVMs:"

SVM_bwList = [0.01, 0.1, 1]
cvFoldList = range(2, 10, 2)

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

    plt.figure(figsize=(13, 8))
    fig = plt.gcf()
    fig.suptitle("SVM results on USPS dataset with hyper-parameter = "+str(SVM_bandwidth), fontsize=15)
    #
    ax = plt.subplot(2, 2, 1)
    plt.plot(cvFoldList, Train_Acc_CV[:, 0], "bo-", label="Linear kernel")
    plt.plot(cvFoldList, Train_Acc_CV[:, 1], "ro-", label="Polynomial kernel")
    plt.plot(cvFoldList, Train_Acc_CV[:, 2], "go-", label="RBF kernel")
    plt.plot(cvFoldList, Train_Acc_CV[:, 3], "ko-", label="Sigmoid kernel")
    plt.ylabel("Training Accuracy")
    # #ax.set_title("Training Accuracy")
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
    # #ax.set_title("Testing Accuracy")
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
    # #ax.set_title("Training Time")
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
    # #ax.set_title("Testing Time")
    plt.grid()
    plt.legend(loc="best",prop={'size':10})
    plt.xlabel("Number of folds in Cross-Validation")
    #
    plt.savefig("USPS_SVM_hyperparm"+str(SVM_bandwidth)+".png", dpi=500)





print "***FINISHED***"





