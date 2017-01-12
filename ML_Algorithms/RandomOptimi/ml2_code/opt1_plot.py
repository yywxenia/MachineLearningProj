import time
from optim_alg import *
import time
import matplotlib.pyplot as pl
import numpy as np



### Ancillary function for printing out schedule information in a nice table:
# --------------------------------------------------------------------------------
# (1) Import data:
flights = dict()
with open('/Users/yywxenia/PycharmProjects/ML_Proj2_final/schedule.txt') as file:
    for line in file:
        origin, dest, depart, arrive, price = line.strip().split(',')
        flights.setdefault((origin, dest), [])
        flights[(origin, dest)].append((depart, arrive, int(price)))

people = [
    ('Seymour', 'BOS'),
    ('Franny', 'DAL'),
    ('Zooey', 'CAK'),
    ('Walt', 'MIA'),
    ('Buddy', 'ORD'),
    ('Les', 'OMA')
]
destination = 'LGA'

domain = [(0, 9)] * (len(people) * 2)

# --------------------------------------------------------------------------------
# (3) Define schedule arrangement cost function:
def getminutes(t):
    x = time.strptime(t, '%H:%M')
    return x[3] * 60 + x[4]


def costfunc(sol):
    totalprice = 0
    latestarrival = 0
    earliestdep = 24 * 60
    temp_out = 0
    temp_ret = 0
    for idx in range(int(len(sol) / 2)):
        origin = people[idx][1]
        outbound = flights[(origin, destination)][int(sol[2 * idx])]
        returnf = flights[(destination, origin)][int(sol[2 * idx + 1])]

        totalprice += (outbound[2] + returnf[2])

        temp_val = getminutes(outbound[1])
        latestarrival = max(latestarrival, temp_val)
        temp_out += temp_val
        temp_val = getminutes(returnf[0])
        earliestdep = min(earliestdep, temp_val)
        temp_ret += temp_val

    totalwait = 0
    totalwait += int(len(sol) / 2) * latestarrival - temp_out
    totalwait += temp_ret - int(len(sol) / 2) * earliestdep
    if latestarrival > earliestdep: totalprice += 50
    return totalprice + totalwait

print "======================================================================================================"
print "Test: GA:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
pops=[10,9,8,7,6,5,4,3,2,1]
elite=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
step=[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
mutate=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

def pop(n):
    return GA(domain, costfunc, popsize=n, step=1, mutprob=0.2, elite=0.2)
def tops(n):
    return GA(domain, costfunc, popsize=50, step=1, mutprob=0.2, elite=n)
def max_step(n):
    return GA(domain, costfunc, popsize=50, step=n, mutprob=0.2, elite=0.2)
def muts(n):
    return GA(domain, costfunc, popsize=50, step=1, mutprob=n, elite=0.2)


###(1) population size:
total_time = []
costs=[]
tr_time=[]
for j in pops:
    for i in range(5):
        start = time.time()
        li = pop(j)
        end = time.time()
        time_elapse1 = (end - start)
        total_time.append(time_elapse1)
        a=costfunc(li)
    costs.append(np.mean(a))
    tr_time.append(np.mean(total_time))

pl.title("Performance of GA: Change Population Size")
pl.subplot(211)
pl.plot(costs, 'g', marker='o', linestyle='--', label='Travel Cost')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(212)
pl.plot(tr_time, 'b', marker='o', linestyle='--', label='Training Time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.show()


###(2) elite percentage:
total_time = []
costs=[]
tr_time=[]
for j in elite:
    for i in range(5):
        start = time.time()
        li = tops(j)
        end = time.time()
        time_elapse1 = (end - start)
        total_time.append(time_elapse1)
        a=costfunc(li)
    costs.append(np.mean(a))
    tr_time.append(np.mean(total_time))

pl.title("Performance of GA: Change Elite Percentage")
pl.subplot(211)
pl.plot(costs, 'g', marker='o', linestyle='--',label='Travel Cost')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(212)
pl.plot(tr_time, 'b', marker='o', linestyle='--',label='Training Time')
pl.legend()
pl.tick_params(axis='x',which='both', bottom='off',top='off',labelbottom='off')

pl.show()


###(3) step percentage:
total_time = []
costs=[]
tr_time=[]
for j in step:
    for i in range(5):
        start = time.time()
        li = max_step(j)
        end = time.time()
        time_elapse1 = (end - start)
        total_time.append(time_elapse1)
        a=costfunc(li)
    costs.append(np.mean(a))
    tr_time.append(np.mean(total_time))

pl.title("Performance of GA: Change Learning Step")
pl.subplot(211)
pl.plot(costs, 'g', marker='o', linestyle='--',label='Travel Cost')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(212)
pl.plot(tr_time, 'b', marker='o', linestyle='--',label='Training Time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.show()
quit()

###(4) mutate percentage:
total_time = []
costs=[]
tr_time=[]
for j in mutate:
    for i in range(5):
        start = time.time()
        li = muts(j)
        end = time.time()
        time_elapse1 = (end - start)
        total_time.append(time_elapse1)
        a=costfunc(li)
    costs.append(np.mean(a))
    tr_time.append(np.mean(total_time))

pl.title("Performance of GA: Change Mutation Percentage")
pl.subplot(211)
pl.plot(costs, 'g', marker='o', linestyle='--',label='Travel Cost')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.subplot(212)
pl.plot(tr_time, 'b', marker='o', linestyle='--',label='Training Time')
pl.legend()
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

pl.show()


