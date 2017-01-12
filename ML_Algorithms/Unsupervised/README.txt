README: Description of python scripts and support files
Yiwei Yan
11/03/2015

Note: libraries used includes not limit to scikit-learn, matplotlib, Pylearn2, etc.
==================================================================
PART 1: SOURCES OF DATA

The datasets we used and imported as .txt format saving at the folder named “Data”.

1. Iris dataset: iris.data.txt
Iris dataset is a very famous classification dataset that consists of 150 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versi-color). Four features were measured from each sample. 

2. USPS digit data: USPS_Digit_Data.txt
This dataset is from USPS handwritten digit database. It includes a popular subset contains 9298 16x16 handwritten digit images in total. The number of classes is 10, and the number of features is 256.

————————————————————————————————————
PART 2: MAIN FUNCTION MODULES

All python scripts used in this report are saved in folder named ”ml3_code". There are totally 11 .py scripts in this folder.

1. ULA.py for implementing all clustering and dimensionality-reduction algorithms. It contains all functions for K-means, EM, PCA, ICA, RP, LDA and Neural Network algorithms. 

2. Create_Dataset.py for dataset setting up.

Please check more parameters details in the python-script files.

————————————————————————————————————
PART 3: RUN THE PYTHON SCRIPTS

All python scripts used in this report are saved in folder named ”ml3_code". There are totally 11 .py scripts in this folder.

1. If you want to implement k-means and EM on Iris and USPS datasets, please run cluster_iris.py and cluster_usps.py.

2. If you want to implement PCA, ICA, RP and LDA implementation on Iris and USPS datasets, please run reduce_iris.py and reduce_usps.py.

3. If you want to implement k-means and EM implementation on new dimensionality-reduced Iris and USPS datasets, please run cluster_reduced_iris.py and cluster_reduced_usps.py.

4. If you want to implement NN training on new dimensionality-reduced USPS datasets, please run USPS_NN_Reduce.py.

5. If you want to implement NN training on new cluster-transformed USPS datasets, please run USPS_NN_Cluster.py.

————————————————————————————————————
PART 4: REFERENCES: GRAPHS AND RESULTS FILES
Please find those files in folder "Support_files".

1. Folder named “Cluster” contains all results .txt files and support graphs for k-means and EM implementation on Iris and USPS datasets.

2. Folder named “DimensionReduction” contains all results .txt files and support graphs for PCA, ICA, RP and LDA implementation on Iris and USPS datasets.

3. Folder named “Cluster_Reduction” contains all results .txt files and support graphs for k-means and EM implementation on new dimensionality-reduced Iris and USPS datasets.

4. Folder named “NN_reduction_USPS” contains all results .txt files and support graphs for NN training on new dimensionality-reduced USPS datasets.

5. Folder named “NN_cluster_USPS” contains all results .txt files and support graphs for NN training on new cluster-transformed USPS datasets.


Thank you!


