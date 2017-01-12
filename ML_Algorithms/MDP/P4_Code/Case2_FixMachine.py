
from MDP_QL_PLOT import *

######################################################################
# (1) Set the base states number for case 2 at 40:
P = fix_machine(40, 0.1, 0.2, 0.3, 0.4, 0.6, 10, 9, 8, 7, 6)[0]
R = fix_machine(40, 0.1, 0.2, 0.3, 0.4, 0.6, 10, 9, 8, 7, 6)[1]

print "P (Case 2)>>>>>> "
for ii in range(len(P)):
    printTexMat(P[ii])
print "R (Case 2)>>>>>> "
printTexMat(R)

######################################################################
#### (1) Plot Value Iteration and Policy Iteration:
#1. P and V iteration times:
a = pi_vi(P, R, 'more', 0, 40)[0]
b = pi_vi(P, R, 'more', 0, 40)[1]
print case_plot1(a, b)

#2. P and V values:
a = pi_vi(P, R, 'more', 0, 40)[4]
b = pi_vi(P, R, 'more', 0, 40)[5]
print case_plot1_value(a, b)

#3. P and V converge time:
at = pi_vi(P, R, 'more', 0, 40)[2]
bt = pi_vi(P, R, 'more', 0, 40)[3]
print plot_case_time(at, bt, 0, ql="without")


# ---------------------------------------------------------------------
#### (2) Plot States change effects: 10 to 100 machines:
print case2__plot("single",0.9, 'pv')

vi_time = case2__plot("single",0.9, 'pv')[2]
pi_time = case2__plot("single",0.9, 'pv')[3]
Machines = range(10, 101, 5)
print conv_pv_time_plot(Machines, vi_time, pi_time)

# ---------------------------------------------------------------------
#### (3) Plot Q-Learning:
## 1> Values comparison with P, V:
base_value = QL_Strategy(P, R, "more", 0, 40)[0]
run_time = QL_Strategy(P, R, "more", 0, 40)[5]
values = QL_Strategy(P, R, "more", 0, 40)[4]
print case_plot2(values, run_time, base_value)

## 2> Values and iter_times changes with greedy factors:
ql_t = plot_ql_time_gammas(P, R)
at = pi_vi(P, R, 'more', 0, 40)[2]
bt = pi_vi(P, R, 'more', 0, 40)[3]
print plot_case_time(at, bt, ql_t, "with")









