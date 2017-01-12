Introduction of python scripts and support files
Yiwei Yan
09/18/2015

Note: libraries used are matplotlib, scikit learn and scikit_neural network
=========================================================================
PART 1: SOURCES OF DATA

 All data we used and imported are .txt format.

1. For algorithm implementation: iris.data.txt
Iris dataset is a very famous classification dataset that consists of 150 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versi-color). Four features were measured from each sample. 
We choose this small and simple dataset to implement our five supervised algorithms: Decision tree, K nearest neighbors, Boosting, SVMs and Neural network.
Please check “iris_intro.txt” in the Support_files folder for more dataset information. The dataset is also in the Support_file folder named iris.data.txt.

2. Two small classification cases: 
(1) bank marketing dataset: bank_full.txt
This dataset is based on "Bank Marketing" UCI dataset (please check the description at: http://archive.ics.uci.edu/ml/datasets/Bank+Marketing), 
which can be approach to predict the success of telemarketing calls for selling bank long-term deposits.The number of Instances is totally 41188 and the number of Attributes are 20 with 1 additional output attribute. 
We used 1000 of this dataset for one small classification case.
Please check “BANK_names_intro.txt” in the Support_files folder for more information. The dataset is also in the Support_file folder named bank_full.txt.

(1) USPS digit handwriting dataset: USPS_Digit_Data.txt
This dataset is from USPS handwritten digit database. It includes a popular subset contains 9298 16x16 handwritten digit images in total. The number of classes is 10, and the number of features is 256.
We used 5000 of this dataset for the other small classification case.
Please check “USPS_intro.txt” in the Support_files folder for more information. The dataset is also in the Support_file folder named USPS_Digit_Data.txt.


————————————————————————————————————
PART 2: FUNCTION MODULES

All algorithms and classification cases implemented by using python and Matlab.

1. Create_Dataset.py for dataset setting up 
There is a python script named “Create_Dataset.py” in Code folder. It contains five functions. Functions are listed as below:
(1) create_dataset(url, i, j, m):
This function helps on read .txt data file and separate it into data X and target data y.
url means url of the dataset, i and j means the start and end columns of data X, and m means the column chosen to be target data y.
(2) PCA_result(n_component, datasets):
This function helps on preprocessing data using PCA to do feature transfermation. n_component means how many dimensions you want to keep in your raw data, and datasets is the raw data.
(3) Shuffle_Data(n_samples):
This function helps to shuffle the raw data index orders. n_samples means the size of raw dataset.
(4) train_test_size(X, y, percent):
This function helps on customizing the size of training dataset and testing dataset.
X means data, y means target data, and percent means how many percentage of the raw data you want to put in the training dataset.
(5) normalize_featDim(dataset): 
This helps do normalization of your dataset.

2. Algorithms.py for algorithm implementing
There is a python script named “Algorithms.py” in Code folder. It contains all functions for running algorithms, testing, and cross-validation.
The function is listed as below:
(1) NN(unit, diffs, XTrain, yTrain, XTest, yTestCorrect): function for running neural networks;
(2) DT_pruning(max_depths, diffs, XTrain, yTrain, XTest, yTestCorrect): function for running decision tree;
(3) Knn(n_neighbor, diffs, XTrain, yTrain, XTest, yTestCorrect): function for running k nearest neighbors;
(4) Boost(max_depths, diffs, XTrain, yTrain, XTest, yTestCorrect): function for running adaBoost;
(5) SVMs(gamma_value, XTrain, yTrain, XTest, yTestCorrect): function for running SVMs;
(6) test_data(X, y, algori_Type, gap, n, m): function for customizing training and testing size for testing under different algorithms.
(7) CVtest(Nfold, X, y, algori_Type, n, m): function for running cross validation testing for different algorithms.

Please check more parameters details in the python scripts.


————————————————————————————————————
PART 3: RUN THE PYTHON SCRIPTS

Please check Code folder for all python scripts.
1. iris data examples for implementing algorithms:
(1) Run Implement_algorithms_IRIS_Final.py file with python to get result of accuracies from K-nn, Decision tree, Boosting and SVMs, and do testing.You can change the value of parameters. 
Besides, run Accs_comparison_Plots.py to see the change of accuracies from different algorithms.

(2) Run Algorithms_DBoundary_Plots.py to visualize iris dataset and get the decision boundaries from different algorithms. 
We used PCA to decrease the dimensions for visualization.

2. CASE 1: Bank marketing case:
Run Case1_BANK_final.py file with python to get result of accuracies from K-nn, Decision tree, Boosting and SVMs algorithms, and do testing.You can change the value of parameters.

3. CASE 2: USPS case:
(1) Run Case2_USPS_final.py file with python to get result of accuracies from K-nn, Decision tree, Boosting and SVMs algorithms, and do testing.You can change the value of parameters.
(2) Run USPS_Graph.py to visualize USPS dataset.


————————————————————————————————————
PART 4: REFERENCES: GRAPHS AND RESULTS FILES

1. GRAPHS in Support_files: 
    This folder contains all result graphs that plotted from this project.
2. RESULTS in Support_files: 
    This folder contains all results information .txt files.


