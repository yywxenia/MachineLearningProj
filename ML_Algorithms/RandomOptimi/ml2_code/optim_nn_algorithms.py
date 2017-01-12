import Create_Dataset
import time
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pyplot as pl
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.optimization.populationbased.ga import GA
from pybrain.optimization.hillclimber import HillClimber
from pybrain.optimization import StochasticHillClimber #This is actually "simulated annealing"


### Helpful function for printing out the layers and weights:
def pesos_conexiones(n):
    for mod in n.modules:
        for conn in n.connections[mod]:
            print conn
            for cc in range(len(conn.params)):
                print conn.whichBuffers(cc), conn.params[cc]

# Create dataset
print "==============================================================================="
iris=Create_Dataset.create_dataset('/Users/yywxenia/PycharmProjects/ML_Proj2_final/iris.data.txt',0,4,4)
n_samples = len(iris['target'])
idx = Create_Dataset.Shuffle_Data(n_samples)
X = iris['data'][idx]
y = iris['target'][idx]

ds = ClassificationDataSet(4,1,nb_classes=3)
for i in range(len(X)):
    a = list(X[i])
    Tlabel = list()
    Tlabel.append(y[i])
    ds.addSample(a, Tlabel)

#Check dataset
print "Sample input dataset:"
print ds['input'][:5]
print "Sample target dataset:"
print ds['target'][:5]
print "Length of the input and target:", len(ds)
print "----------------------------------------------------"

#Split training and testing set
traindata, testdata=ds.splitWithProportion(0.7)
print "training data length:", len(traindata)
print "testing data length:", len(testdata)
print "-----------------------------------------------------"

# Encode classes with one output neuron per class:
traindata._convertToOneOfMany()
testdata._convertToOneOfMany()
print "Sample encoded target data at training dataset: "
print traindata['target'][:5]
print "Sample Class:"
print traindata['class'][:5]
print "------------------------------------------------------"

# Create a neural network and show the parameter information:
net = buildNetwork(4,3,3, outclass=SoftmaxLayer, bias=False)
print 'by finding good weights for this (simple) network:'
print "NN structure:", net
print 'which has', len(net.params), 'trainable parameters.'

# Impliment different optimization algorithms on NN
print "==============================================================================="
print "(0) Using Backprob~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
### Train model with backprop method:
net_bp = buildNetwork(4,3,3, outclass=SoftmaxLayer, bias=False)
trainer = BackpropTrainer(net_bp, dataset=traindata, momentum=0.1, verbose=True, weightdecay=.01)
final_training_error = list()
final_testing_error = list()
train_t = list()
test_t = list()

# max_iter_num = trainer.totalepochs
max_iter_num = 100
for itr in range(0, max_iter_num):
    start = time.time()
    trainer.trainOnDataset(traindata, 1)
    end = time.time()
    time_elapse1 = (end - start)
    train_t.append(time_elapse1)
    current_training_err = percentError(net_bp.activateOnDataset(traindata).argmax(axis=1), traindata['class'])
    start2 = time.time()
    current_testing_err = percentError(net_bp.activateOnDataset(testdata).argmax(axis=1),  testdata['class'])
    end2 = time.time()
    time_elapse1 = (end2 - start2)
    test_t.append(time_elapse1)
    final_training_error.append(current_training_err)
    final_testing_error.append (current_testing_err)
print 'Training error of BackPropogation: ', min(final_training_error), final_training_error
print 'Testing error of BackPropogation: ', min(final_testing_error), final_testing_error
print 'Training time of BackPropogation: ', sum(train_t)
print 'Testing time of BackPropogation: ', sum(test_t)

pl.plot(final_training_error, 'b', label='Training error of BP')
pl.plot(final_testing_error, 'r', label='Testing error of BP')
pl.title("Accuracy of BP on iris NN")
pl.legend()
pl.show()



print "\n"
print "Implementation: "
print '(1) Hill Climb~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
### (2) Hill Climb:
mse_hc = list()
mse_hct = list()
train_t = list()
test_t = list()
net_hc = buildNetwork(4,3,3, outclass=SoftmaxLayer, bias=False)
hc = HillClimber(traindata.evaluateModuleMSE, net_hc, minimize=True)

maxEval_hc = 500
for i in range(maxEval_hc):
    start = time.time()
    Temp_hc = hc.learn(0)
    end = time.time()
    time_elapse1 = (end - start)
    train_t.append(time_elapse1)
    mse_hc.append(Temp_hc[1])
    net_hc = Temp_hc[0]
    # build testing model
    hct = HillClimber(testdata.evaluateModuleMSE, net_hc, minimize=True)
    start2 = time.time()
    testing_error = hct.learn(-1)[1]           # -1 means not train the test data.
    end2 = time.time()
    time_elapse1 = (end2 - start2)
    test_t.append(time_elapse1)
    # print "testing error: ", testing_error
    mse_hct.append(testing_error)
print 'Training error of hill climb:', min(mse_hc), mse_hc
print 'Testing error of hill climb: ', min(mse_hct), mse_hct
print 'Training time of Hill Climb: ', sum(train_t)
print 'Testing time of Hill Climb: ', sum(test_t)

pl.plot(mse_hc, 'b', label='Training error of hill climb')
pl.plot(mse_hct, 'r', label='Testing error of hill climb')
pl.title("Accuracy of RHC on iris NN")
pl.legend()
pl.show()

print '(2) Simulated Annealing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
### (3) Simulated Annealing:
mse_sa = list()
mse_sat = list()
train_t = list()
test_t = list()
net_sa = buildNetwork(4,3,3, outclass=SoftmaxLayer, bias=False)
sa = StochasticHillClimber(traindata.evaluateModuleMSE, net_sa, minimize=True, maxLearningSteps=1,temperature=1000)
maxEval_sa = 100

for i in range(maxEval_sa):
    start = time.time()
    Temp_sa = sa.learn(0)
    end = time.time()
    time_elapse1 = (end - start)
    train_t.append(time_elapse1)
    mse_sa.append(Temp_sa[1])
    net_sa = Temp_sa[0]
    # build testing model
    sat = StochasticHillClimber(testdata.evaluateModuleMSE, net_sa, minimize=True, maxLearningSteps=1, temperature=1000) # Minimize cost
    start2 = time.time()
    testing_error = sat.learn(-1)[1]             # -1 means not train the test data.
    end2 = time.time()
    time_elapse1 = (end2 - start2)
    test_t.append(time_elapse1)
    mse_sat.append(testing_error)
print 'Training error of simulated annealing: ', min(mse_sa), mse_sa
print 'Testing error of simulated annealing: ', min(mse_sat), mse_sat
print 'Training time of simulated annealing: ', sum(train_t)
print 'Testing time of simulated annealing: ', sum(test_t)

pl.plot(mse_sa, 'b', label='Training error of simulated annealing')
pl.plot(mse_sat, 'r', label='Testing error of simulated annealing')
pl.title("Accuracy of SA on iris NN")
pl.legend()
pl.show()

print '(3) GA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
### GA optimization:
mse_ga = list()
mse_gat = list()
train_t = list()
test_t = list()

net_ga = buildNetwork(4,3,3, outclass=SoftmaxLayer, bias=False)
ga = GA(traindata.evaluateModuleMSE, net_ga, minimize=True, mutationProb=0.2)
maxEval_ga = 100

for i in range(maxEval_ga):
    start = time.time()
    Temp_ga = ga.learn(0)
    end = time.time()
    time_elapse1 = (end - start)
    train_t.append(time_elapse1)
    mse_ga.append(Temp_ga[1])
    net_ga = Temp_ga[0]

    # build testing model
    gat = GA(testdata.evaluateModuleMSE, net_ga, minimize=True, mutationProb=0.2)
    start2 = time.time()
    testing_error = gat.learn(0)[1]
    end2 = time.time()
    time_elapse1 = (end2 - start2)
    test_t.append(time_elapse1)
    mse_gat.append(testing_error)
print 'Training error of GA:', min(mse_ga), mse_ga
print 'Testing error of GA:', min(mse_gat), mse_gat
print 'Training time of GA: ', sum(train_t)
print 'Testing time of GA: ', sum(test_t)

pl.plot(mse_ga, 'b', label='Training error of GA')
pl.plot(mse_gat, 'r', label='Testing error of GA')
pl.title("Accuracy of GA on iris NN")
pl.legend()
pl.show()

print "-------------------------------***********FINISHED************-------------------------------------"





