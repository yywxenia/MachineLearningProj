import Create_Dataset
import time
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
import matplotlib.pyplot as pl
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.optimization.populationbased.ga import GA
from pybrain.optimization import StochasticHillClimber
import numpy as np


### Helpful function for printing out the layers and weights:
def pesos_conexiones(n):
    for mod in n.modules:
        for conn in n.connections[mod]:
            print conn
            for cc in range(len(conn.params)):
                print conn.whichBuffers(cc), conn.params[cc]

# Create dataset
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

#Split training and testing set
traindata, testdata=ds.splitWithProportion(0.7)
# Encode classes with one output neuron per class:
traindata._convertToOneOfMany()
testdata._convertToOneOfMany()
# Create a neural network and show the parameter information:
net = buildNetwork(4,3,3, outclass=SoftmaxLayer, bias=False)
#
# Implement different optimization algorithms on NN
print " For SA: ==================================================================================="
mse_sa = list()
mse_sat = list()
train_t = list()
test_t = list()
net_sa = buildNetwork(4,3,3, outclass=SoftmaxLayer, bias=False)

## (1) Situation: change temperature
def temp(t):
    return StochasticHillClimber(traindata.evaluateModuleMSE, net_sa, minimize=True,
                                 maxLearningSteps=1,temperature=t)
def temp_test(t):
    return StochasticHillClimber(testdata.evaluateModuleMSE, net_sa, minimize=True,
                                 maxLearningSteps=1,temperature=t)

temps=[10000, 5000, 2500, 1800, 1000, 800, 100, 50, 10, 5, 1, 0.1]
maxEval_sa = 2
av_sa=[]
av_sat=[]
tr_t=[]
te_t=[]

for j in temps:
    for i in range(maxEval_sa):
        start = time.time()
        Temp_sa = temp(j).learn(0)
        end = time.time()
        time_elapse1 = (end - start)
        train_t.append(time_elapse1)
        mse_sa.append(Temp_sa[1])
        net_sa = Temp_sa[0]

        sat = temp_test(j)
        start2 = time.time()
        testing_error = sat.learn(-1)[1]
        end2 = time.time()
        time_elapse2 = (end2 - start2)
        test_t.append(time_elapse2)
        mse_sat.append(testing_error)

    av_sa.append(np.mean(mse_sa))
    av_sat.append(np.mean(mse_sat))
    tr_t.append(np.mean(train_t))
    te_t.append(np.mean(test_t))
print 'Training error:', av_sa
print 'Testing error:', av_sat
print 'Trainig time:',tr_t
print 'Testing time:', te_t

pl.title("Performance of SA: Change Temperature")
pl.subplot(411)
pl.plot(av_sa, 'g', label='Training error')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(412)
pl.plot(av_sat, 'b', label='Testing error')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(413)
pl.plot(tr_t, 'g--', label='Training time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(414)
pl.plot(te_t, 'b--', label='Testing time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.show()

print "---------------------------------------------"
# Situation: change learning steps
mse_sa = list()
mse_sat = list()
train_t = list()
test_t = list()
def step(s):
    return StochasticHillClimber(traindata.evaluateModuleMSE, net_sa, minimize=True,
                                       maxLearningSteps=s, temperature=1000) # Minimize cost
def step_test(s):
    return StochasticHillClimber(testdata.evaluateModuleMSE, net_sa, minimize=True,
                                       maxLearningSteps=s, temperature=1000) # Minimize cost

steps = [1000, 800, 600, 400, 200, 100, 50, 30, 10, 5, 3, 1, 0.5]
maxEval_sa = 2
av_sa=[]
av_sat=[]
tr_t=[]
te_t=[]

for j in steps:
    for i in range(maxEval_sa):
        start = time.time()
        Temp_sa = step(j).learn(0)
        end = time.time()
        time_elapse1 = (end - start)
        train_t.append(time_elapse1)
        mse_sa.append(Temp_sa[1])
        net_sa = Temp_sa[0]

        sat = step_test(j)
        start2 = time.time()
        testing_error = sat.learn(-1)[1]
        end2 = time.time()
        time_elapse2 = (end2 - start2)
        test_t.append(time_elapse2)
        mse_sat.append(testing_error)
    av_sa.append(np.mean(mse_sa))
    av_sat.append(np.mean(mse_sat))
    tr_t.append(np.mean(train_t))
    te_t.append(np.mean(test_t))

print 'Training error:', av_sa
print 'Testing error:', av_sat
print 'Trainig time:',tr_t
print 'Testing time:', te_t

pl.title("Performance of SA: Change Learning Step")
pl.subplot(411)
pl.plot(av_sa, 'g', label='Training error')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(412)
pl.plot(av_sat, 'b', label='Testing error')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(413)
pl.plot(tr_t, 'g--', label='Training time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(414)
pl.plot(te_t, 'b--', label='Testing time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.show()
quit()
print " For GA: ==================================================================================="
### GA optimization:
mse_ga = list()
mse_gat = list()
train_t = list()
test_t = list()

net_ga = buildNetwork(4,3,3, outclass=SoftmaxLayer, bias=False)
### (1) mutation
def mut(m):
    return GA(traindata.evaluateModuleMSE, net_ga, minimize=True, mutationProb=m)
def mut_test(m):
    return GA(testdata.evaluateModuleMSE, net_ga, minimize=True, mutationProb=m)
maxEval_ga = 2
mutations = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
av_ga=[]
av_gat=[]
tr_t=[]
te_t=[]

for j in mutations:
    for i in range(maxEval_ga):
        start = time.time()
        Temp_ga = mut(j).learn(0)
        end = time.time()
        time_elapse1 = (end - start)
        train_t.append(time_elapse1)
        mse_ga.append(Temp_ga[1])
        net_ga = Temp_ga[0]

        gat = mut_test(j)
        start2 = time.time()
        testing_error = gat.learn(0)[1]
        end2 = time.time()
        time_elapse2 = (end2 - start2)
        test_t.append(time_elapse2)
        mse_gat.append(testing_error)
    tr_t.append(np.mean(train_t))
    te_t.append(np.mean(test_t))
    av_ga.append(np.mean(mse_ga))
    av_gat.append(np.mean(mse_gat))
print 'Training error: ', av_ga
print 'Testing error:', av_gat
print 'Trainig time:',tr_t
print 'Testing time:', te_t

pl.title("Performance of GA: Change Mutation-prob")
pl.subplot(411)
pl.plot(av_ga, 'g', label='Training error')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(412)
pl.plot(av_gat, 'b', label='Testing error')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(413)
pl.plot(tr_t, 'g--', label='Training time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(414)
pl.plot(te_t, 'b--', label='Testing time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.show()



## GA top propotion changes:
mse_ga = list()
mse_gat = list()
train_t = list()
test_t = list()

net_ga = buildNetwork(4,3,3, outclass=SoftmaxLayer, bias=False)

def topp(m):
    return GA(traindata.evaluateModuleMSE, net_ga, minimize=True, mutationProb=0.2, topProportion=m)
def topp_test(m):
    return GA(testdata.evaluateModuleMSE, net_ga, minimize=True, mutationProb=0.2, topProportion=m)

maxEval_ga = 2
proportions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1]
av_ga=[]
av_gat=[]
tr_t=[]
te_t=[]

for j in proportions:
    for i in range(maxEval_ga):
        start = time.time()
        Temp_ga = topp(j).learn(0)
        end = time.time()
        time_elapse1 = (end - start)
        train_t.append(time_elapse1)
        mse_ga.append(Temp_ga[1])
        net_ga = Temp_ga[0]

        gat = topp_test(j)
        start2 = time.time()
        testing_error = gat.learn(0)[1]
        end2 = time.time()
        time_elapse2 = (end2 - start2)
        test_t.append(time_elapse2)
        mse_gat.append(testing_error)

    tr_t.append(np.mean(train_t))
    te_t.append(np.mean(test_t))
    av_ga.append(np.mean(mse_ga))
    av_gat.append(np.mean(mse_gat))
print 'Training error: ', av_ga
print 'Testing error:', av_gat
print 'Trainig time:',tr_t
print 'Testing time:', te_t

pl.title("Performance of GA: Change Top Proportion")
pl.subplot(411)
pl.plot(av_ga, 'g', label='Training error')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(412)
pl.plot(av_gat, 'b', label='Testing error')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(413)
pl.plot(tr_t, 'g--', label='Training time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(414)
pl.plot(te_t, 'b--', label='Testing time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.show()


print "-------------------------------***********FINISHED************-------------------------------------"






