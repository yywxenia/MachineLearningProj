
### Implement optimization algorithms: reference from Programming Collective Intelligence
### =============================================================================================
import random
import math


#-----------------------------------------------------------------------------------------
# Hill climb
def hill_climb(domain, cost_func):
## Generates a random list of numbers in the given domain to create the initial solution
    # Create a random solution
    sol = [random.randint(domain[idx][0], domain[idx][1]) for idx in range(len(domain))]   # build a random pop
    # main loop
    while True:
        neighbors = []
        for j in range( len(domain) ):
            if sol[j] > domain[j][0]:
                #print(sol[j],domain[j][0])
                neighbors.append(sol[0:j] + [sol[j]-1] + sol[j+1:])
            if sol[j] < domain[j][1]:
                neighbors.append(sol[0:j] + [sol[j]+1] + sol[j+1:])
        # See what the best solution amongst the neighbors is
        current = cost_func(sol)
        best = current
        for j in range( len(neighbors) ):
            cost = cost_func( neighbors[j] )
            if cost < best:
                best = cost
                sol = neighbors[j]
        # If there's no improvement, then we've reached the top
        if best == current:
            break
    return sol


#-----------------------------------------------------------------------------------------
# Simulated Annealing:
def anneal_optim(domain, costfunc, T, cool, step):
    vec = [ random.randint(domain[i][0], domain[i][1])
            for i in range(len(domain)) ]
    while T > 0.1:
        i = random.randint(0, len(domain)-1)
        dir = random.randint(-step, step)
        vecb = vec[:]
        vecb[i] += dir

        if vecb[i] < domain[i][0]:
            vecb[i] = domain[i][0]
        elif vecb[i] > domain[i][1]:
            vecb[i] = domain[i][1]

        ea = costfunc(vec) # lower cost
        eb = costfunc(vecb) # higher cost
        p = pow(math.e, (-eb - ea)/T) # The probability of a higher-cost solution being accepted:
        if (eb < ea or random.random() < p):
            vec = vecb
        T = T*cool          # Decrease the temperature
    return vec


#-----------------------------------------------------------------------------------------
## GA:
def GA(domain, costfunc, popsize, step, mutprob, elite, maxiter=100):

    def mutate(vec):
        i = random.randint(0, len(domain)-1)
        if random.random() < 0.5 and vec[i] > domain[i][0]:
            return vec[0:i] + [vec[i] - step] + vec[i+1: ]
        elif vec[i] < domain[i][1]:
            return vec[0:i] + [vec[i] + step] + vec[i+1: ]
        else:
            return vec

    def crossover(r1, r2):
        i = random.randint(0, len(domain)-2)
        return r1[0:i] + r2[i:]

    # Initially create a set of random solutions known as the "population"
    pop = []
    for i in range(popsize):
        vec = [ random.randint(domain[idx][0], domain[idx][1])
              for idx in range(len(domain)) ]
        pop.append(vec)

    # How many winners from each generation
    topelite = int( elite * popsize)  # Winner of every generation == wining rate * population size
    # main loop
    for i in range(maxiter):
        scores = [ (costfunc(v),v) for v in pop ]
        scores.sort()
        ranked = [ v for (s,v) in scores ]

        # Start with the pure winners
        pop = ranked[0:topelite]

        # Add mutated and bred forms of the winners
        while len(pop) < popsize:
            if random.random() < mutprob:
                # mutation
                idx = random.randint(0, topelite)
                pop.append( mutate(ranked[idx]) )
            else:
                # crossover
                idx1 = random.randint(0, topelite)
                idx2 = random.randint(0, topelite)
                pop.append( crossover(ranked[idx1], ranked[idx2]))
        # #Print current best score
        # print scores[0][0]
    return scores[0][1]
