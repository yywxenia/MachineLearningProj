print " Optimization 1 - Group Travel: Planning a trip for a group of people from different locations to the same place"
print "Reference from 'Programming Collective Intelligence"
print "\n"
print "======================================================================================================"
import time
from optim_alg import *
from mimicry.mimic import Mimic

### Ancillary function for printing out schedule information in a nice table:
# --------------------------------------------------------------------------------
def printschedule(r):
    for idx in range(int(len(r) / 2)):
        name = people[idx][0]
        origin = people[idx][1]
        out = flights[(origin, destination)][int(r[2 * idx])]
        ret = flights[(destination, origin)][int(r[2 * idx + 1])]
        print '%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name, origin, out[0], out[1], out[2], ret[0], ret[1], ret[2])

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

print "(0) Selectable total difections of flights: ", len(flights.items())
print "\n"
for key, value in flights.items():
    print "(origin location, destination location): ", key
    print "(departure time, arrival time, price): ", value
print "\n"
print "(1) Total number of people: ", len(people), type(people)

# --------------------------------------------------------------------------------
# (2)
s = [1, 4, 3, 2, 7, 3, 6, 3, 2, 4, 5, 3]  # 0 is the first flight of the day, 1 is the second, and so on.
print "\n"
print "(2) Randomly give a list of flights a person choose to take (outbound & return):"
print s
print "\n"
print "(3) A table of the above random-given flight infos:"
print printschedule(s)

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


# ~~~~~for MIMIC cost function minimization (The mimicry originally is maximize fitness function) ~~~~~
def costfunc2(sol):
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
    return -(totalprice + totalwait)


print "(4) Schedule arrangement's total cost from the above choices is: ", costfunc(s)
print "\n"

# --------------------------------------------------------------------------------
print "======================================================================================================"
print "(5) Implement optimization algorithms on this problem: "
domain = [(0, 9)] * (len(people) * 2)
print "Domain: ", domain, type(domain)
print "\n"
#
# print "Test: hill climb~~~~~~~~~~~~~~~~~~~~~~~~~"
# total_time = []
# for i in range(20):
#     start = time.time()
#     li = hill_climb(domain, costfunc)
#     end = time.time()
#     time_elapse1 = (end - start)
#     total_time.append(time_elapse1)
# print "Total time spend on Hill Climb optimization:", sum(total_time)
# print "Hill Climbing: ", hill_climb(domain, costfunc)
# print "Total cost after hill climb optimization:", (costfunc(li))
# print "Schedule detail: "
# print printschedule(li)
#
# print "\n"
# print "Test: simulated annealing:~~~~~~~~~~~~~~~"
# total_time = []
# for i in range(20):
#     start = time.time()
#     li = anneal_optim(domain, costfunc, T=1000, cool=0.95, step=1)
#     end = time.time()
#     time_elapse1 = (end - start)
#     total_time.append(time_elapse1)
# print "Total time spend on simulated annealing optimization:", sum(total_time)
# print "Simulated annealing: ", li
# print "Total cost after simulated annealing optimization:", (costfunc(li))
# print "Schedule detail: "
# print printschedule(li)

print "\n"
print "Test: GA:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
total_time = []
for i in range(2):
    start = time.time()
    li = GA(domain, costfunc, popsize=10, step=1, mutprob=0.2, elite=0.2)
    end = time.time()
    time_elapse1 = (end - start)
    total_time.append(time_elapse1)
print "Total time spend on simulated annealing optimization:", sum(total_time)
print "Genetic algorithm: ", li
print "Total cost after GA optimization:", (costfunc(li))
print "Schedule detail: "
print printschedule(li)
print "\n"
#

total_time = []
for i in range(2):
    start = time.time()
    li = GA(domain, costfunc, popsize=10, step=0.9, mutprob=0.2, elite=0.2)
    end = time.time()
    time_elapse1 = (end - start)
    total_time.append(time_elapse1)
print "Total time spend on simulated annealing optimization:", sum(total_time)
print "Genetic algorithm: ", li
print "Total cost after GA optimization:", (costfunc(li))
print "Schedule detail: "
print printschedule(li)
print "\n"
# print "Test: MIMIC:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
# total_time = []
# m = Mimic(domain, costfunc, samples=1000)
# for i in range(20):
#     start = time.time()
#     mm = m.fit()
#     l_mm = mm.tolist()
#     print l_mm
#
#     li = l_mm[-1]  # THE BEST COMBINATION
#     result = costfunc(li)
#     end = time.time()
#     time_elapse1 = (end - start)
#     total_time.append(time_elapse1)
#
# print "Total time spend on MIMIC optimization:", sum(total_time)
# print "MIMIC: ", GA(domain, costfunc)
# print "Total cost after MIMIC optimization:", (costfunc(li))
# print "Schedule detail: "
# print printschedule(li)

