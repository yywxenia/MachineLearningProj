__author__ = 'yywxenia'
print " Optimization 3 -  "
print "\n"
print "======================================================================================================"
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

###(2)
def costfunc2(sol):
        return -(-exp(-1) + (sol[0]-100)*(sol[0]-102) - exp(-1) + (sol[1]-100)*(sol[1]-102))


#--------------------------------------------------------------------------------
print "Implement optimization algorithms on this problem: "
domain = [(50, 200)]*2
print domain
print "Domain: ", domain, type(domain)
print "\n"


print "Test: hill climb~~~~~~~~~~~~~~~~~~~~~~~~~"
cost_opt = 100
li_opt = [0, 0]
total_time=[]
for i in range(10):
    start=time.time()
    li = hill_climb(domain, costfunc)
    current_cost = costfunc(li)
    if current_cost < cost_opt:
        cost_opt = current_cost
        li_opt = li
    end = time.time()
    time_elapse1 = (end - start)
    total_time.append(time_elapse1)
print "Total time spend on hill climb optimization:", sum(total_time)
print "Hill Climbing: ", li_opt
print "Total cost after hill climb optimization:", cost_opt
print "\n"


print "Test: simulated annealing:~~~~~~~~~~~~~~~"
cost_opt = 100
li_opt = [0, 0]
total_time=[]
for i in range(10):
    start = time.time()
    li = anneal_optim(domain, costfunc, T=10, cool=0.99, step=1)
    current_cost = costfunc(li)
    if current_cost < cost_opt:
        cost_opt = current_cost
        li_opt = li
    end = time.time()
    time_elapse1 = (end - start)
    total_time.append(time_elapse1)
print "Total time spend on simulated annealing optimization:", sum(total_time)
print "Simulated An: ", li_opt
print "Total cost after SA optimization:", cost_opt
print "\n"

quit()
print "Test: GA:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cost_opt = 100
li_opt = [0, 0]
total_time = []
for i in range(10):
    start = time.time()
    li = GA(domain, costfunc, popsize=10, step=3, mutprob=0.5, elite=0.5)
    current_cost = costfunc(li)
    if current_cost < cost_opt:
        cost_opt = current_cost
        li_opt = li
    end = time.time()
    time_elapse1 = (end - start)
    total_time.append(time_elapse1)
print "Total time spend on GA optimization:", sum(total_time)
print "GA: ", li_opt
print "Total cost after GA optimization:", cost_opt


print "Test: MIMIC:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cost_opt = 10000
li_opt = [0, 0]
m = Mimic(domain, costfunc2, samples=10000)
mm=m.fit()

for i in range(10):
    start = time.time()
    mm = m.fit()
    print "mm:", mm
    l_mm = mm.tolist()
    print "lmm:", l_mm
    li = l_mm[0]       # THE BEST COMBINATION
    current_cost = costfunc2(li)
    result = -1*(costfunc2(li))
    if result < cost_opt:
        cost_opt = result
        li_opt = li
    end = time.time()
    time_elapse1 = (end - start)
    total_time.append(time_elapse1)
print "Total time spend on MIMIC optimization:", sum(total_time)
print "MIMIC: ", li_opt
print "Total cost after MIMIC optimization:", cost_opt