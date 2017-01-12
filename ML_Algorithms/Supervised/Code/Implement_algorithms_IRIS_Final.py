print(__doc__)
import Create_Dataset
from Algorithms import CVtest
from Algorithms import test_data

iris=Create_Dataset.create_dataset('/Users/yywxenia/Desktop/iris.data.txt',0,4,4)

### Shuffle the original iris dataset:
##--------------------------------------------------------------------------------
n_samples = len(iris['target'])
idx = Create_Dataset.Shuffle_Data(n_samples)

X = iris['data'][idx]
y = iris['target'][idx]


#(3) Impliment different algorithms on the Bank dataset
print "==============================================================================="
print "  1st: Using SVMs:"
SVM_result = test_data(X, y, "SVMs", 0.2, 0, 0)
print "----------------------------------------------"
SVM_CV_result = CVtest(4, X, y, "SVMs",0.2, 0)

print "==============================================================================="
print "  2ed: Using K Nearest Neighbors:"
Knn_result = test_data(X, y, "Knn", 0.8, 5, 1)
print "----------------------------------------------"
Knn_CV_result = CVtest(8, X, y, "Knn", 5, 1)


print "==============================================================================="
print "  3rd: Using Decision Tree:"
DT_Result = test_data(X, y, "DT_pruning", 0.6, 5, 1)

print "----------------------------------------"
DT_CV_Result = CVtest(4, X, y, "DT_pruning", 5, 1)
# #
# #
print "=============================================================================="
print "  4th: Using AdaBoosting:"
Boost_Result = test_data(X, y, "Boost", 0.8, 5, 1)

print "----------------------------------------"
Boost_CV_Result = CVtest(4, X, y, "Boost", 5, 1)

print "***FINISHED***"

