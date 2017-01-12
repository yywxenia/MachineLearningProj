print(__doc__)
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import cross_validation
from pylab import *
import time
import math
from Create_Dataset import train_test_size
from sknn.mlp import Classifier, Layer


### Algorithms
##------------------------------------------------------------------------------------------------
###(0) Nueral Network:
def NN(unit, diffs, XTrain, yTrain, XTest, yTestCorrect):
    #unit means the number of nodes.
    DT_Test_Depth_acc=[]
    DT_Train_Depth_acc=[]
    max_acc = -10

    Time_train = []
    Time_test = []

    for unit in range(diffs, unit+1, diffs):
        clf = Classifier(layers=[Layer("Sigmoid", units=unit), Layer("Sigmoid", units=unit), Layer("Sigmoid")],
                         learning_rate=0.1, n_iter=30)

        start = time.time()
        clf = clf.fit(XTrain, yTrain)
        end = time.time()
        time_elapse1 = (end - start)
        Time_train.append(time_elapse1)
        # print "-->time used on training model: ", time_elapse1

        Train_predicted = clf.predict(XTrain)
        Train_accs = accuracy_score(yTrain, Train_predicted) #Training accuracy
        DT_Train_Depth_acc.append(Train_accs)

        start2 = time.time()
        Test_predicted = clf.predict(XTest)
        end2 = time.time()
        time_elapse2 = (end2 - start2)
        Time_test.append(time_elapse2)
        # print "---->time used on testing model: ", time_elapse2

        Test_accs = accuracy_score(yTestCorrect, Test_predicted) #Testing accuracy.
        DT_Test_Depth_acc.append(Test_accs)

        if max_acc <= Test_accs:
            max_acc = Test_accs

    Depth_Num=[]
    for i in range(0, int(math.floor(unit/diffs))):
        if DT_Test_Depth_acc[i] == max_acc:
            Depth_Num.append(i + 1)

    return DT_Train_Depth_acc, DT_Test_Depth_acc, max_acc, Depth_Num, Time_train, Time_test


###(1) Decision Tree:
def DT_pruning(max_depths, diffs, XTrain, yTrain, XTest, yTestCorrect):
    # max_depths means the largest depths of DT you give, and diffs means the changing of depths in iterations.
    DT_Test_Depth_acc=[]
    DT_Train_Depth_acc=[]
    max_acc = -10

    Time_train = []
    Time_test = []
    for i in range(diffs, max_depths+1, diffs):
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=i)

        start = time.time()
        clf = clf.fit(XTrain, yTrain)
        end = time.time()
        time_elapse1 = (end - start)
        Time_train.append(time_elapse1)
        # print "-->time used on training model: ", time_elapse1

        Train_predicted = clf.predict(XTrain)
        Train_accs = accuracy_score(yTrain, Train_predicted) #Training accuracy
        DT_Train_Depth_acc.append(Train_accs)

        start2 = time.time()
        Test_predicted = clf.predict(XTest)
        end2 = time.time()
        time_elapse2 = (end2 - start2)
        Time_test.append(time_elapse2)
        # print "---->time used on testing model: ", time_elapse2

        Test_accs = accuracy_score(yTestCorrect, Test_predicted) #Testing accuracy.

        DT_Test_Depth_acc.append(Test_accs)

        if max_acc <= Test_accs:
            max_acc = Test_accs

    Depth_Num=[]
    for i in range(0, int(math.floor(max_depths/diffs))):
        if DT_Test_Depth_acc[i] == max_acc:
            Depth_Num.append(i + 1)

    return DT_Train_Depth_acc, DT_Test_Depth_acc, max_acc, Depth_Num, Time_train, Time_test


### (2) K nearest neighbors:
def Knn(n_neighbor, diffs, XTrain, yTrain, XTest, yTestCorrect):
    # n_neighbors means the largest number of neighbors you give, and diffs means the changing of neighbor bumbers in iterations.
    K_T_acc=[]
    K_Tr_acc=[]
    max_acc = -10

    Time_train = []
    Time_test = []
    for i in range(1, n_neighbor+1, diffs): # iterate numbers of neighbors from 1 to n_neighbor
        # clf = neighbors.KNeighborsClassifier(i, weights='distance') #set weights to be 'distance'
        clf = neighbors.KNeighborsClassifier(i, p=2)

        start = time.time()
        clf = clf.fit(XTrain, yTrain)
        end = time.time()
        time_elapse1 = (end - start)
        Time_train.append(time_elapse1)
        # print "-->time used on training model: ", time_elapse1

        Train_predicted = clf.predict(XTrain)
        Tr_accs = accuracy_score(yTrain, Train_predicted) #Training accuracy
        K_Tr_acc.append(Tr_accs)

        start2 = time.time()
        predicted = clf.predict(XTest)
        end2 = time.time()
        time_elapse2 = (end2 - start2)
        Time_test.append(time_elapse2)
        # print "---->time used on testing model: ", time_elapse2

        T_accs = accuracy_score(yTestCorrect, predicted) #Testing accuracy score
        K_T_acc.append(T_accs) # save accuracy scores from different k neighbors

        if max_acc <= T_accs:
            max_acc = T_accs # gain the max accuracy value for each iteration

    max_k=[]
    for i in range(0, int(math.floor(n_neighbor/diffs))):
        if K_T_acc[i] == max_acc:
            max_k.append(i+1)

    return K_Tr_acc, K_T_acc, max_acc, max_k, Time_train, Time_test


### (3) AdaBoosting:
def Boost(max_depths, diffs, XTrain, yTrain, XTest, yTestCorrect):
# max_depths means the largest number of depths you give, and diffs means the changing of depths in iterations.
    BT_Depth_acc=[]
    BTr_Depth_acc=[]
    max_acc = -10

    Time_train = []
    Time_test = []
    for i in range(1, max_depths+1, diffs):
        clf_1 = DecisionTreeClassifier(max_depth = max_depths)
        clf_2 = AdaBoostClassifier(clf_1)

        start = time.time()
        clf = clf_2.fit(XTrain, yTrain)
        end = time.time()
        time_elapse1 = (end - start)
        Time_train.append(time_elapse1)
        # print "-->time used on training model: ", time_elapse1

        Train_predicted = clf.predict(XTrain)
        Tr_accs = accuracy_score(yTrain, Train_predicted) #Training accuracy
        BTr_Depth_acc.append(Tr_accs)

        start2 = time.time()
        predicted = clf.predict(XTest)
        end2 = time.time()
        time_elapse2 = (end2 - start2)
        Time_test.append(time_elapse2)
        # print "---->time used on testing model: ", time_elapse2
        BT_accs = accuracy_score(yTestCorrect, predicted) #Testing accuracy
        BT_Depth_acc.append(BT_accs)

        if max_acc <= BT_accs:
            max_acc = BT_accs

    Depth_Num=[]
    for i in range(0, int(math.floor(max_depths/diffs))):
        if BT_Depth_acc[i] == max_acc:
            Depth_Num.append(i+1)

    return BTr_Depth_acc, BT_Depth_acc, max_acc, Depth_Num, Time_train, Time_test


### (4) SVMs:
def SVMs(gamma_value, XTrain, yTrain, XTest, yTestCorrect):
    accuracy=[]
    Tr_accuracy=[]

    Time_train = []
    Time_test = []
    for kernels in ('linear', 'poly', 'rbf', 'sigmoid'):
        clf = svm.SVC(kernel=kernels, gamma=gamma_value)

        start = time.time()
        clf = clf.fit(XTrain, yTrain)
        end = time.time()
        time_elapse1 = (end - start)
        Time_train.append(time_elapse1)
        # print "-->time used on training model: ", time_elapse1

        Train_predicted = clf.predict(XTrain)
        Tr_accs = accuracy_score(yTrain, Train_predicted) #Training accuracy
        Tr_accuracy.append(Tr_accs)

        start2 = time.time()
        predicted = clf.predict(XTest)
        end2 = time.time()
        time_elapse2 = (end2 - start2)
        Time_test.append(time_elapse2)
        # print "---->time used on testing model: ", time_elapse2

        accs = accuracy_score(yTestCorrect, predicted) # Testing accuracy
        accuracy.append(accs)

    return Tr_accuracy, accuracy, Time_train, Time_test



### (5). Customize training dataset size testing
##------------------------------------------------------------------------------------------------
def test_data(X, y, algori_Type, gap, n, m):
    X_train, y_train, X_test, y_test = train_test_size(X, y, gap)

     ## When using Neural Network:
    if algori_Type == "NN":
        training_acc, testing_acc, test_max_ac, Depth_N, Time_train, Time_test = NN(n, m, X_train, y_train, X_test, y_test)
        # n is the number of max depth, m is the changes of depths in iterations

        print "  The training accuracies: ", training_acc
        print "  The testing accuracies:", testing_acc
        print "   The max testing accuracy: ", max(testing_acc)


    ## When using Decision Tree:
    if algori_Type == "DT_pruning":
        training_acc, testing_acc, test_max_ac, Depth_N, Time_train, Time_test = DT_pruning(n, m, X_train, y_train, X_test, y_test)
        # n is the number of max depth, m is the changes of depths in iterations

        print "  The training accuracies: ", training_acc
        print "  The testing accuracies:", testing_acc
        print "   The max testing accuracy: ", max(testing_acc)

    ## When using K nearest neighbors:
    elif algori_Type == "Knn":
        K_Tr_acc, K_T_acc, acc_max, k_max, Time_train, Time_test = Knn(n, m, X_train, y_train, X_test, y_test)

        print "  The training accuracies: ", K_Tr_acc
        print "  The testing accuracies: ", K_T_acc
        print "   The max testing accuracy: ", max(K_T_acc)


      ## When using SVMs:
    elif algori_Type=="SVMs":
        Tr_accuracy, accuracy, Time_train, Time_test = SVMs(n, X_train, y_train, X_test, y_test)
        # n means gamma_vaule

        print "   The training accuracy from linear-kernel, poly-kernel, rbf-kernel, sigmoid-kernel: ", Tr_accuracy
        print "   The testing accuracy from linear-kernel, poly-kernel, rbf-kernel, sigmoid-kernel: ", accuracy
        print "   The max testing accuracy: ", max(accuracy)

      ## When using AdaBoost:
    elif algori_Type == "Boost":
        BTr_Depth_acc, BT_Depth_acc, max_acc, Depth_Num, Time_train, Time_test = Boost(n, m, X_train, y_train, X_test, y_test)

        print "  The training accuracies:", BTr_Depth_acc
        print "  The testing accuracies:", BT_Depth_acc
        print "  The max testing accuracy: ", max(BT_Depth_acc)


###(6) Cross Validation Test
##------------------------------------------------------------------------------------------------
def CVtest(Nfold, X, y, algori_Type, n, m):
    kf = cross_validation.KFold(1000, n_folds = Nfold) # If you want to run all data, change 1000 to be len(y)

    k_all = []
    CV_max_acc_k_all = []
    CV_max_acc = []
    Depth_Num_all = []
    CV_max_acc_DT_all = []


    TrainAccCollection=[]
    TestAccCollection=[]
    TrainTimeAccCollection=[]
    TestTimeCollection=[]
    for train_index, test_index in kf:
        # Make training and testing datasets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if algori_Type == "NN":
            training_acc, testing_acc, test_max_ac, Depth_N, Time_train, Time_test = NN(n, m, X_train, y_train, X_test, y_test)

            TrainAccCollection.append(training_acc)
            TestAccCollection.append(testing_acc)
            TrainTimeAccCollection.append(Time_train)
            TestTimeCollection.append(Time_test)

            # n is the number of max depth
            print "  The training accuracies from the cross validation: ", training_acc
            print "  The testing accuracies from the cross validation: ", testing_acc
            print "  The max accuracy from the cross validation: ", test_max_ac
            print "  Corresponding DT depths: ",  Depth_N

            CV_max_acc.append(test_max_ac)
            #DT_depth_selected.append(Depth_N[np.round(len(Depth_N)/2)])
            for i in range(len(Depth_N)):
                Depth_Num_all.append(Depth_N[i])
                CV_max_acc_DT_all.append((test_max_ac))
            print "CV max accuracies from iterations: ", CV_max_acc


        ## When using decision tree:
        elif algori_Type == "DT_pruning":
            training_acc, testing_acc, test_max_ac, Depth_N, Time_train, Time_test = DT_pruning(n, m, X_train, y_train, X_test, y_test)

            TrainAccCollection.append(training_acc)
            TestAccCollection.append(testing_acc)
            TrainTimeAccCollection.append(Time_train)
            TestTimeCollection.append(Time_test)

            # n is the number of max depth
            print "  The training accuracies from the cross validation: ", training_acc
            print "  The testing accuracies from the cross validation: ", testing_acc
            print "  The max accuracy from the cross validation: ", test_max_ac
            print "  Corresponding DT depths: ",  Depth_N

            CV_max_acc.append(test_max_ac)
            #DT_depth_selected.append(Depth_N[np.round(len(Depth_N)/2)])
            for i in range(len(Depth_N)):
                Depth_Num_all.append(Depth_N[i])
                CV_max_acc_DT_all.append((test_max_ac))
            print "CV max accuracies from iterations: ", CV_max_acc


        ## When using K Nearest Neighbors:
        elif algori_Type == "Knn":
            K_Tr_acc, K_T_acc, acc_max, k_max, Time_train, Time_test = Knn(n, m, X_train, y_train, X_test, y_test)
            TrainAccCollection.append(K_Tr_acc)
            TestAccCollection.append(K_T_acc)
            TrainTimeAccCollection.append(Time_train)
            TestTimeCollection.append(Time_test)

            print "  The training accuracies from the cross validation: ", K_Tr_acc
            print "  The testing accuracies from the cross validation: ", K_T_acc
            print "  The max accuracy: ", acc_max
            print "  Corresponding k to mac accuracy: ",  k_max

            CV_max_acc.append(acc_max)

            for i in range(len(k_max)):
                k_all.append(k_max[i])
                CV_max_acc_k_all.append(acc_max)

            print "CV max accuracies from iterations: ", CV_max_acc
            #print "k_selected: ", k_selected


        ## When using SVMs:
        elif algori_Type=="SVMs":
            Tr_accuracy, accuracy, Time_train, Time_test = SVMs(n, X_train, y_train, X_test, y_test)
            TrainAccCollection.append(Tr_accuracy)
            TestAccCollection.append(accuracy)
            TrainTimeAccCollection.append(Time_train)
            TestTimeCollection.append(Time_test)

            print "   The testing accuracy from linear-kernel, poly-kernel, rbf-kernel at this iteration: ", accuracy
            print "   The training accuracy from linear-kernel, poly-kernel, rbf-kernel at this iteration: ", Tr_accuracy
            print "   The max test accuracy at this iteration: ", max(accuracy)


        ## When using Boosting:
        elif algori_Type == "Boost":

            BTr_Depth_acc, BT_Depth_acc, max_acc, Depth_Num, Time_train, Time_test = Boost(n, m, X_train, y_train, X_test, y_test)
            TrainAccCollection.append(BTr_Depth_acc)
            TestAccCollection.append(BT_Depth_acc)
            TrainTimeAccCollection.append(Time_train)
            TestTimeCollection.append(Time_test)

            print "  The training accuracies from the cross validation: ", BTr_Depth_acc
            print "  The testing accuracies from the cross validation: ", BT_Depth_acc
            print "  The max accuracy from the cross validation: ", max_acc
            print "  The Corresponding DT depths: ",  Depth_Num

            CV_max_acc.append(max_acc)
            for i in range(len(Depth_Num)):
                Depth_Num_all.append(Depth_Num[i])
                CV_max_acc_DT_all.append((max_acc))

            print "Max accuracies from CV iterations: ", CV_max_acc

        else:
            raise ValueError('In Algorithm.py: [Fatal] Unrecognized algorithm type!')



    # Get mean of TrainAccCollection
    assert(len(TrainAccCollection) == Nfold)


    lenEachItr = len(TrainAccCollection[0])
    meanTrainAcc = TrainAccCollection[0]
    meanTestAcc  = TestAccCollection[0]
    meanTrainTime = TrainTimeAccCollection[0]
    meanTestTime  = TestTimeCollection[0]

    for i in range(1, Nfold):
        for j in range(lenEachItr):
            meanTrainAcc[j] += TrainAccCollection[i][j]
            meanTestAcc[j]  += TestAccCollection[i][j]
            meanTrainTime[j]+= TrainTimeAccCollection[i][j]
            meanTestTime[j] += TestTimeCollection[i][j]

    for j in range(lenEachItr):
        meanTrainAcc[j] = float(meanTrainAcc[j])/float(Nfold)
        meanTestAcc[j] = float(meanTestAcc[j])/float(Nfold)
        meanTrainTime[j] = float(meanTrainTime[j])/float(Nfold)
        meanTestTime[j] = float(meanTestTime[j])/float(Nfold)

    print "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    print ">>>>>>>>>>> Summary >>>>>>>>>>"
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    print "Mean training accuracy: ", meanTrainAcc
    print "Mean testing accuracy: ", meanTestAcc
    print "Mean training time: ", meanTrainTime
    print "Mean testing time: ", meanTestTime
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    return meanTrainAcc, meanTestAcc, meanTrainTime, meanTestTime

