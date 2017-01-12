import mdptoolbox
import numpy as np
import matplotlib.pyplot as plt
import time


######################################################################
#### Print Matrix for LaTex:
def printTexMat(a):
    print "\\begin{bmatrix} "
    for i in range(a.shape[0]):
        line = ""
        for j in range(a.shape[1]-1):
            line += str(a[i][j])+ "  &  "
        line +=  str(a[i][a.shape[1]-1])+ " \\\\ "
        print line
    print "\\end{bmatrix} "


######################################################################
#### Create Datasets for This Report:

# ===================================================
### Case 1:
def graduate_path0():
    P1 = np.zeros((2, 4, 4))
    P1[0, 0, 2] = 1
    P1[0, 1, 0] = 1
    P1[0, 2, 3] = 1
    P1[0, 3, 3] = 1
    P1[1, 0, 1] = 1
    P1[1, 1, 1] = 1
    P1[1, 2, 1] = 1
    P1[1, 3, 1] = 1

    # Rewards:
    R1 = np.zeros((4, 2))
    R1[2, 0] = 50
    R1[3, 0] = 30
    R1[0, 1] = 100
    R1[1, 1] = 30
    R1[3, 1] = -50
    return P1, R1


def graduate_path1():
    P2 = np.zeros((2, 4, 4))
    P2[0, 0, 2] = 1
    P2[0, 1, 0] = 0.9
    P2[0, 2, 2] = 0.1
    P2[0, 3, 3] = 1
    P2[0, 1, 1] = 0.1
    P2[0, 2, 3] = 0.9

    P2[1, 0, 1] = 1
    P2[1, 1, 1] = 1
    P2[1, 2, 1] = 0.9
    P2[1, 2, 2] = 0.1
    P2[1, 3, 1] = 0.9
    P2[1, 3, 3] = 0.1

    # Rewards:
    R2 = np.zeros((4, 2))
    R2[2, 0] = 50
    R2[3, 0] = 30
    R2[0, 1] = 100
    R2[1, 1] = 30
    R2[3, 1] = -50

    return P2, R2

# ===================================================
### Case 2:
def fix_machine(S, p1, p2, p3, p4, p5, r1, r2, r3, r4, r5):
# Action1: No maintenance: 90% chance that machines are depreciated.

    P = np.zeros((5, S, S))                   #'p1' means the percent that machines can remain the current conditions.
    P[0, :, :] = p1 * np.diag(np.ones(S), 0)
    P[0, S - 1, S - 1] = 1
    for i in range (S-1):
        P[0, i, i+1] = 1-p1
        P[0, i, i+1] = 1 - p1

# Action2: level1_maintenance
   #'p2' means after taking the lowest level of maintenance, the % that machines can remain the current conditions.

    P[1, :, :] = (p2 * np.diag(np.ones(S), 0))
    pp = 1-p2-(1-(p2+0.2))+0.1  #0.2
    P[1, 1, 0] = pp
    P[1, 0, 0] = p2+0.2
    P[1, 0, 1] = 1-(p2+0.2)
    P[1, S - 1, S - 1] = (1-(p2+0.2))+p2-0.1
    for i in range (1,S-1):
        P[1, i, i+1] =1-(p2+0.2)-0.1
        P[1, i+1, i] = pp*0.7
        P[1, i+1, i-1] = pp*0.3

# Action3: level2_maintenance
    P[2, :, :] = (p3 * np.diag(np.ones(S), 0))
    pp = 1-p3-(1-(p3+0.2))+0.1
    P[2, 1, 0] = pp
    P[2, 0, 0] = p3+0.2
    P[2, 0, 1] = 1-(p3+0.2)
    P[2, S - 1, S - 1] = (1-(p3+0.2))+p3-0.1
    for i in range (1,S-1):
        P[2, i, i+1] = 1-(p3+0.2)-0.1
        P[2, i+1, i] = pp*0.6
        P[2, i+1, i-1] = pp*0.4

# Action4: level3_maintenance
    P[3, :, :] = (p4 * np.diag(np.ones(S), 0))
    pp = 1-p4-(1-(p4+0.2))+0.1
    P[3, 1, 0] = pp
    P[3, 0, 0] = p4+0.2
    P[3, 0, 1] = 1-(p4+0.2)
    P[3, S - 1, S - 1] = (1-(p4+0.2))+p4-0.1
    for i in range (1,S-1):
        P[3, i, i+1] = 1-(p4+0.2)-0.1
        P[3, i+1, i] = pp*0.6
        P[3, i+1, i-1] = pp*0.4

# Action5: level4_maintenance_the best maintenance
    P[4, :, :] = ((1-p5) * np.diag(np.ones(S), 0))
    P[4, 1, 0] = p5
    P[4, 0, 0] = 1
    for i in range (1,S-1):
        P[4, i+1, i] = (p5)*0.6
        P[4, i+1, i-1] = (p5)*0.4

# Rewards:
    R = np.zeros((S, 5))  #There aare totally 5 actions
    r = [r1,r2,r3,r4,r5]
  #(1) Do not maintain the machine, the depreciation at each level
    for k in range(S):
        R[k, 0]= r1-0.7*k-0
        R[k, 1]= r2-0.6*k-1
        R[k, 2]= r3-0.4*k-2
        R[k, 3]= r4-0.2*k-4
        R[k, 4]= r5-0.1*k-7
    return P, R



######################################################################
#### Compute iteration time:
def pi_vi(p, r, ga_num, ga, states_num):
    if ga_num =="more":   #(1)
        i_t = []
        vi_time=[]
        v_v=[]
        p_t = []
        pi_time=[]
        p_v=[]
        for gamma_v in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
            print "Iterations with GAMMA >>>>>>>>>>", gamma_v

            start = time.time()
            pi = mdptoolbox.mdp.PolicyIteration(p, r, gamma_v, skip_check=True)
            print "Policy-iteration ~~~~~~ \n", pi.policy
            pi.setVerbose()
            pi.run()
            end = time.time()
            time_elapse1 = (end - start)
            pi_time.append(time_elapse1)
            print "-->time used on pi: ", time_elapse1
            p_t.append(pi.iter)
            pv = sum(pi.V)/states_num
            p_v.append(pv)

            start = time.time()
            vi = mdptoolbox.mdp.ValueIteration(p, r, gamma_v, skip_check=True)
            print "Value-iteration ~~~~~~ \n", vi.policy
            vi.setVerbose()
            vi.run()
            end = time.time()
            time_elapse1 = (end - start)
            vi_time.append(time_elapse1)
            print "-->time used on vi: ", time_elapse1
            i_t.append(vi.iter)
            vv = sum(vi.V)/states_num
            v_v.append(vv)
            print "--------------------------------"

        print "(1) Total iteration times for value-iteration with different gamma values: \n", i_t, "\n"
        print "(2) Total iteration times for policy-iteration with different gamma values: \n", p_t, "\n"
        print "(3) Value-iteration converge time with different gamma values: \n", vi_time, "\n"
        print "(4) Policy-iteration converge time with different gamma values: \n", pi_time, "\n"
        print "(5) Value-iteration values with different gamma values: \n", v_v, "\n"
        print "(6) Policy-iteration values with different gamma values: \n", p_v, "\n"
        print "=================================================="
        return i_t, p_t, vi_time, pi_time, v_v, p_v

    elif ga_num =="single":   #(2)
        i_t = []
        vi_time=[]
        v_v=[]
        p_t = []
        pi_time=[]
        p_v=[]

        start = time.time()
        pi = mdptoolbox.mdp.PolicyIteration(p, r, ga, skip_check=True)
        print "Policy-iteration ~~~~~~ \n", pi.policy
        pi.setVerbose()
        pi.run()
        end = time.time()
        time_elapse1 = (end - start)
        pi_time.append(time_elapse1)
        p_t.append(pi.iter)
        pv = sum(pi.V)/(pi.iter)
        p_v.append(pv)

        start = time.time()
        vi = mdptoolbox.mdp.ValueIteration(p, r, ga, skip_check=True)
        print "Value-iteration ~~~~~~ \n", vi.policy
        vi.setVerbose()
        vi.run()
        end = time.time()
        time_elapse1 = (end - start)
        vi_time.append(time_elapse1)
        i_t.append(vi.iter)
        vv = sum(vi.V)/(vi.iter)
        v_v.append(vv)
        print "--------------------------------------------------"
        print "(1) Total iteration times for value-iteration at GAMMA",ga,": \n", i_t, "\n"
        print "(2) Total iteration times for policy-iteration at GAMMA",ga,": \n", p_t, "\n"
        print "(3) Value-iteration converge time at GAMMA",ga,": \n", vi_time, "\n"
        print "(4) Policy-iteration converge time at GAMMA",ga,": \n", pi_time, "\n"
        print "(5) Value-iteration values at GAMMA",ga,": \n", v_v, "\n"
        print "(6) Policy-iteration values at GAMMA",ga,": \n", p_v, "\n"
        print "=================================================="
        return i_t, p_t, vi_time, pi_time, v_v, p_v


######################################################################
#### Q-learning:
def QL_Strategy(p, r, gf_num, greedy_factor, states_n):
    v_time=[]
    p_time=[]
    ql_list = []
    ql_time = []

    start = time.time()
    pi = mdptoolbox.mdp.PolicyIteration(p, r, 0.9, skip_check=True)
    pi.run()
    end = time.time()
    time_elapse1 = (end - start)
    p_time.append(time_elapse1)
    print "REAL Values from policy-iteration: "
    print pi.V
    print "REAL policies from policy-iteration:"
    print pi.policy
    pi_v = sum(pi.V)/states_n
    print "-----------------"
    vi = mdptoolbox.mdp.ValueIteration(p, r, 0.9, skip_check=True)
    vi.run()
    end = time.time()
    time_elapse1 = (end - start)
    v_time.append(time_elapse1)
    print "REAL Values from value-iteration: "
    print vi.V
    print "REAL policies from value-iteration:"
    print vi.policy
    vi_v = sum(vi.V)/states_n
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if gf_num =="more":
        for ii in range(0, 21, 2):
            np.random.seed(0)
            fac = float(ii) / 10.0
            print "Greedy factor >>>>>>>>", fac
            start = time.time()
            ql = mdptoolbox.mdp.QLearning(p, r, 0.9, fac)
            ql.run()
            end = time.time()
            time_elapse1 = (end - start)
            ql_time.append(time_elapse1)
            ql.Q
            print "Q: \n", ql.Q
            print "Selected Policy: \n", ql.policy
            print "Value: \n", ql.V
            qlv = np.mean(ql.V)
            ql_list.append(qlv)
        print "(1) Q-Learning values with different greedy factors: \n", ql_list, "\n"
        print "(2) Q-learning converge time with different greedy factors: \n", ql_time, "\n"
        print "============================================="
        #return ql_list, ql_time

    elif gf_num =="single":
        sql_time=[]

        print "Greedy factor >>>>>>>>", greedy_factor
        start = time.time()
        ql = mdptoolbox.mdp.QLearning(p, r, 0.9, greedy_factor)
        ql.run()
        end = time.time()
        time_elapse1 = (end - start)
        sql_time.append(time_elapse1)
        ql.Q
        print "Q: \n", ql.Q
        print "Selected Policy: \n", ql.policy
        print "Value: \n", ql.V
        qlv = np.mean(ql.V)
        print "(1) Q-Learning values with greedy factors", greedy_factor, ": \n", qlv, "\n"
        print "(2) Q-learning converge time with greedy factors", greedy_factor, ": \n", ql_time, "\n"
        #return qlv, ql_time
        print "============================================="
    return pi_v, vi_v, v_time, p_time, ql_list, ql_time, qlv, ql_time


######################################################################
#### Plot Q-learning time changes through gamma value changes:
def plot_ql_time_gammas(p, r):
    ql_time=[]
    print "Greedy factor >>>>>>>>", 1
    for gamma_v in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
        print "GAMMA Value is: >>>>>>>>>>", gamma_v
        start = time.time()
        ql = mdptoolbox.mdp.QLearning(p, r, gamma_v, 1)
        ql.run()
        end = time.time()
        time_elapse1 = (end - start)
        ql_time.append(time_elapse1)
        ql.Q
        print "Q: \n", ql.Q
        print "Selected Policy: \n", ql.policy
        print "Value: \n", ql.V
    return ql_time



######################################################################
#### Plot for Policy and Value Iterations:
def case_plot1(a, b):
    gamma_vs = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    ### iteration times graph
    plt.subplot(211)
    plt.plot(gamma_vs, a, 'go-', label='Value Iteration')
    plt.ylabel('Value Iteration Times')
    plt.subplots_adjust(bottom=0.1)
    plt.margins(0.1)
    plt.legend(loc='upper left')
    plt.grid()
    # plt.title("Iteration times on different gamma")

    plt.subplot(212)
    plt.plot(gamma_vs, b, 'ro-', label='Policy Iteration')
    plt.ylabel('Policy Iteration Times')
    plt.xlabel('Gamma')
    plt.subplots_adjust(bottom=0.1)
    plt.margins(0.05)
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()


def case_plot1_value(a, b):
    gamma_vs = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    plt.plot(gamma_vs, a, 'ko-', label='Value Iteration')
    plt.plot(gamma_vs, b, linewidth=1.5, color='b', label='Policy Iteration')
    plt.ylabel('Values')
    plt.xlabel('Gamma')
    plt.subplots_adjust(bottom=0.1)
    plt.margins(0.1)
    plt.legend(loc='best')
    plt.grid()
    plt.show()

######################################################################
#### Plot for converge time comparison:
def plot_case_time(at, bt, qlt, ql):
    if ql=="with":
        gamma_vs = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        plt.plot(gamma_vs, qlt, 'bo-', label='Q-Learning Time')
        plt.plot(gamma_vs, at, 'mo-', label='Value-Iteration')
        plt.plot(gamma_vs, bt, 'yo-', label='Policy-Iteration')
        plt.ylabel('Converge Time')
        plt.xlabel('Gamma')
        plt.subplots_adjust(bottom=0.1)
        plt.margins(0.1)
        plt.legend(loc='best')
        plt.grid()
        plt.show()
    elif ql=="without":
        gamma_vs = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        plt.plot(gamma_vs, at, 'mo-', label='Value Iteration')
        plt.plot(gamma_vs, bt, 'yo-', label='Policy Iteration')
        plt.ylabel('Converge Time')
        plt.xlabel('Gamma')
        plt.subplots_adjust(bottom=0.1)
        plt.margins(0.1)
        plt.legend(loc='best')
        plt.grid()
        plt.show()


######################################################################
# Plot for Q_learning:
def case_plot2(c, run_time, base):
    greedy_factors = [0, .2, .4, .6, .8, 1, 1.2, 1.4, 1.6, 1.8, 2]
    base_line = [base] * len(greedy_factors)

    ### ql_strategies graph:
    plt.subplot(211)
    plt.plot(greedy_factors, base_line, linewidth=1.5, color="r", linestyle='--', label='Real Value')
    plt.plot(greedy_factors, c,  "bo-", label='Q-Learning Value')
    plt.ylabel('Iteration values')
    plt.margins(0.05)
    plt.legend(loc='best')
    plt.grid()

    plt.subplot(212)
    plt.plot(greedy_factors, run_time,  "mo-", label='Q-Learning Time')
    plt.xlabel('Greedy factors of strategies')
    plt.ylabel('Convergence time')
    plt.margins(0.05)
    plt.legend(loc='best')
    plt.grid()

    plt.show()


######################################################################
##### For states number effects: (Case 2)

### (1) Plot states affection graph:
def case2__plot(ga_num, ga, algor):
    Machines = range(10, 101, 5)
    iter_v=[]
    iter_p=[]
    P_time=[]
    V_time=[]
    if algor == "pv":
        for state_num in Machines:
            P_n = fix_machine(state_num, 0.1, 0.2, 0.3, 0.4, 0.6, 10, 9, 8, 7, 6)[0]
            R_n = fix_machine(state_num, 0.1, 0.2, 0.3, 0.4, 0.6, 10, 9, 8, 7, 6)[1]
            result = pi_vi(P_n, R_n, ga_num, ga)
            iter_v.append((result[0]))
            iter_p.append((result[1]))
            P_time.append(result[3])
            V_time.append(result[2])

        plt.subplot(211)
        plt.plot(Machines, iter_v,  "ro-", label='Value Iteration')
        plt.ylabel('Iterations at gamma 0.9')
        plt.margins(0.05)
        plt.subplots_adjust(bottom=0.1)
        plt.legend(loc="lower right")
        plt.grid()

        plt.subplot(212)
        plt.plot(Machines, iter_p, "go-",  label='Policy Iteration')
        plt.xlabel('Number of Machines (States)')
        plt.ylabel('Iterations at gamma 0.9')
        plt.margins(0.05)
        plt.subplots_adjust(bottom=0.1)
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()
        return iter_v, iter_p, P_time, V_time


  ### (2) Plot Converge time graph:
def conv_pv_time_plot(Machines, V_time, P_time):
    plt.plot(Machines, V_time,  "yo-", label='Value iteration')
    plt.plot(Machines, P_time,  "mo-", label='Policy iteration')
    plt.ylabel('Convergence time with gamma 0.9')
    plt.xlabel('Number of Machines (States)')
    plt.margins(0.05)
    plt.subplots_adjust(bottom=0.1)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
