import matplotlib.pyplot as pl
import numpy as np
import time
from optim_alg import *
from mimicry.mimic import Mimic
from math import *
from random import *

#--------------------------------------------------------------------------------
# (1) Define cost function:
#cost_function for minimizing
def costfunc2(sol):
    return -10 * sin(5 * sol[0]) - 7 * cos(4 * sol[1]) + 17 # For MIMIC using fitness function: maximization


##samples: Number of samples to generate from the distribution each iteration
##percentile: Percentile of the distribution to keep after each iteration, default is 0.90

#--------------------------------------------------------------------------------
print "Implement optimization algorithms on this problem: "
domain = [(-50, 50)]*2
# m = Mimic(domain, costfunc2, samples=10000, percentile=0.9)
# mm = m.fit()

print "Test: MIMIC:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cost_opt = 1000
li_opt = [0, 0]
total_time=[]

sample=[12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000]
perc=[ 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
def samples(n):
    return Mimic(domain, costfunc2, samples=n, percentile=0.9)
def percs(n):
    return Mimic(domain, costfunc2, samples=5000, percentile=n)


 costs=[]
 tr_time=[]
 for j in sample:
     for i in range(5):
         start = time.time()
         mm = samples(j).fit()
         l_mm = mm.tolist()
         li = l_mm[0]       # THE BEST COMBINATION
         current_cost = costfunc2(li)
         result = -1*(costfunc2(li)-17)+17
         if result < cost_opt:
             cost_opt = result
             li_opt = li
         end = time.time()
         time_elapse1 = (end - start)
         total_time.append(time_elapse1)
         c=cost_opt
     costs.append(np.mean(c))
     tr_time.append(np.mean(total_time))

 pl.title("Performance of MIMIC: Change Sample Size")
 pl.subplot(211)
 pl.plot(costs, 'g', marker='o', linestyle='--', label='Cost')
 pl.legend()
 pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

 pl.subplot(212)
 pl.plot(tr_time, 'b', marker='o', linestyle='--', label='Train Time')
 pl.legend()
 pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

 pl.show()



### (2) percentile:

costs=[]
tr_time=[]
for j in perc:
    for i in range(1):
        start = time.time()
        mm = percs(j).fit()
        l_mm = mm.tolist()
        li = l_mm[0]       # THE BEST COMBINATION
        current_cost = costfunc2(li)
        result = -1*(costfunc2(li)-17)+17
        if result < cost_opt:
            cost_opt = result
            li_opt = li
        end = time.time()
        time_elapse1 = (end - start)
        total_time.append(time_elapse1)
        c=cost_opt
    costs.append(np.mean(c))
    tr_time.append(np.mean(total_time))

pl.title("Performance of MIMIC: Change Percentile")
pl.subplot(211)
pl.plot(costs, 'g', marker='o', linestyle='--', label='Cost')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(212)
pl.plot(tr_time, 'b', marker='o', linestyle='--', label='Train Time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.show()
