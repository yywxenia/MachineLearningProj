print " Optimization 2 - Search the minimum value of a function"
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
    return 10 * sin(5 * sol[0]) + 7 * cos(4 * sol[1]) + 17  #'+17' ensure the result to be positive, no other meaning
def costfunc2(sol):
    return -10 * sin(5 * sol[0]) - 7 * cos(4 * sol[1]) + 17 # For MIMIC using fitness function: maximization

#--------------------------------------------------------------------------------
print "Implement optimization algorithms on this problem: "
domain = [(-50, 50)]*2
print "Domain: ", domain, type(domain)
print "\n"

print "Test: hill climb~~~~~~~~~~~~~~~~~~~~~~~~~"
cost_opt = 1000
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
cost_opt = 1000
li_opt = [0, 0]
total_time=[]
for i in range(10):
    start = time.time()
    li = anneal_optim(domain, costfunc, T=1000, cool=0.95, step=1)
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


print "Test: GA:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cost_opt = 1000
li_opt = [0, 0]
total_time=[]
for i in range(10):
    start = time.time()
    li = GA(domain, costfunc, popsize=50, step=1, mutprob=0.2, elite=0.2)
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
cost_opt = 1000
li_opt = [0, 0]
total_time=[]
m = Mimic(domain, costfunc2, samples=10000, percentile=0.7)

for i in range(10):
    start=time.time()
    mm = m.fit()
    l_mm = mm.tolist()
    li = l_mm[0]       # THE BEST COMBINATION
    current_cost = costfunc2(li)
    result = -1*(costfunc2(li)-17)+17
    if result < cost_opt:
        cost_opt = result
        li_opt = li
    # print "result: ", result
    # print "li: ", li
    # print "cost_opt so far: ", cost_opt
    # print "---"
    end = time.time()
    time_elapse1 = (end - start)
    total_time.append(time_elapse1)
print "Total time spend on MIMIC optimization:", sum(total_time)
print "MIMIC: ", li_opt
print "Total cost after MIMIC optimization:", cost_opt

print "-------------------------------***********FINISHED************-------------------------------------"

