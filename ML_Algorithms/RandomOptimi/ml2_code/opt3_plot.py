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
def costfunc(sol):
    return -exp(-1) + (sol[0]-100)*(sol[0]-102) - exp(-1) + (sol[1]-100)*(sol[1]-102)

#--------------------------------------------------------------------------------
print "Implement optimization algorithms on this problem: "
domain = [(40, 200)]*2
# mm = m.fit()

print "Test: simulated annealing:~~~~~~~~~~~~~~~"
cost_opt = 100
li_opt = [0, 0]
total_time=[]

Ts=[1000, 700, 400, 100, 70, 40, 10, 7, 4, 1] #decrease
cooling=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99] #increase
lsteps=[12,11,10,9,8,7,6,5,4,3,2,1]

#anneal_optim(domain, costfunc, T=5, cool=0.99, step=1)

def T(n):
    return anneal_optim(domain, costfunc, T=n, cool=0.99, step=1)
def Cools(n):
    return anneal_optim(domain, costfunc, T=10, cool=n, step=1)
def l_steps(n):
    return anneal_optim(domain, costfunc, T=10, cool=0.99, step=n)


 costs=[]
 tr_time=[]
 for j in Ts:
     for i in range(20):
         start = time.time()
         li = T(j)
         current_cost = costfunc(li)
         if current_cost < cost_opt:
             cost_opt = current_cost
             li_opt = li
         end = time.time()
         time_elapse1 = (end - start)
         total_time.append(time_elapse1)
         c=cost_opt
     costs.append(np.mean(c))
     tr_time.append(np.mean(total_time))

 pl.title("Performance of SA: Change Temperature")
 pl.subplot(211)
 pl.plot(costs, 'g', marker='o', linestyle='--', label='Cost')
 pl.legend()
 pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

 pl.subplot(212)
 pl.plot(tr_time, 'b', marker='o', linestyle='--', label='Train Time')
 pl.legend()
 pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

 pl.show()


 #(2)
 costs=[]
 tr_time=[]
 for j in cooling:
     for i in range(20):
         start = time.time()
         li = Cools(j)
         current_cost = costfunc(li)
         if current_cost < cost_opt:
             cost_opt = current_cost
             li_opt = li
         end = time.time()
         time_elapse1 = (end - start)
         total_time.append(time_elapse1)
         c=cost_opt
     costs.append(np.mean(c))
     tr_time.append(np.mean(total_time))

 pl.title("Performance of SA: Change Cooling Level")
 pl.subplot(211)
 pl.plot(costs, 'g', marker='o', linestyle='--', label='Cost')
 pl.legend()
 pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

 pl.subplot(212)
 pl.plot(tr_time, 'b', marker='o', linestyle='--', label='Train Time')
 pl.legend()
 pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

 pl.show()

##(3)
costs=[]
tr_time=[]
for j in lsteps:
    for i in range(20):
        start = time.time()
        li = l_steps(j)
        current_cost = costfunc(li)
        if current_cost < cost_opt:
            cost_opt = current_cost
            li_opt = li
        end = time.time()
        time_elapse1 = (end - start)
        total_time.append(time_elapse1)
        c=cost_opt
    costs.append(np.mean(c))
    tr_time.append(np.mean(total_time))

pl.title("Performance of SA: Change Learning Steps")
pl.subplot(211)
pl.plot(costs, 'g', marker='o', linestyle='--', label='Cost')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(212)
pl.plot(tr_time, 'b', marker='o', linestyle='--', label='Train Time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.show()









