README: Description of python scripts and support files
Yiwei Yan
10/17/2015

Note: libraries used are matplotlib, mimicry, numpy, and PyBrain, etc.
==================================================================
PART 1: SOURCES OF DATA

 All data we used and imported as .txt format.

1. For Neural Network: iris.data.txt
Iris dataset is a very famous classification dataset that consists of 150 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versi-color). Four features were measured from each sample. 
We choose this small and simple dataset to implement operation algorithms on Neural Network.

2. For the first operation problem: schedule.txt 
This dataset is downloaded from the book "Programming Collective Intelligence" by Toby Segaran, chapter 5.

————————————————————————————————————
PART 2: FUNCTION MODULES

All algorithms and classification cases implemented by using python. All the code are saved in folder "ml2_code".

1. Create_Dataset.py for dataset setting up and shufflering if necessary.

2. optim_alg.py for randomized operation algorithms implementation. It contains all functions for RHC, SA, GA, MIMIC algorithms. 

3. optim_nn_algorithms.py for Neural Network optimization. This script is for implement Neural Network based on library PyBrain. It runs Backprob and all four randomized operation algorithms for iris data Neural Network.

Please check more parameters details in the python-script files.

————————————————————————————————————
PART 3: RUN THE PYTHON SCRIPTS

All these python scripts are in folder "ml2_code".

1. If you want to check the results, please run optim_nn_algorithms.py for operation on Neural Network, optim_q1_GA.py for operation problem 1, optim_q2_MIMIC.py for operation problem 2, and optim_q3_SA_2Dimpy.py for operation problem 3.

2. If you want to check the fighres appears in the report, please run nn_plot.py for operation on Neural Network, opt1_plot.py for operation problem 1, opt2_plot.py for operation problem 2, and opt3_plot.py for operation problem 3.

————————————————————————————————————
PART 4: REFERENCES: GRAPHS AND RESULTS FILES
Please find those files in folder "pics and results".

1. All results.txt files for optimized Nueral Network (optim_nn_results_500), operation problem 1(opt1.txt), operation problem 2(opt2.txt), and poperation problem 3(opt3.txt) are in folder "pics and results".

2. All pictures used in the report can be found in this fold "pics and results" as well.


Thank you!


