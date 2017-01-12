
from MDP_QL_PLOT import *

#########################################################################
#### (1) Create P,R matrices: Derterministic
P1 = graduate_path0()[0]
R1 = graduate_path0()[1]

print "P1 (Case 1)>>>>>> "
print P1
print "R1 (Case 1)>>>>>> "
print R1

# ---------------------------------------------------------------------
#### (2) Create P,R matrices: Non-derterministic
P2 = graduate_path1()[0]
R2 = graduate_path1()[1]

print "P2 (Case 1)>>>>>> "
print P2
print "R2 (Case 1)>>>>>> "
print R2


######################################################################
#### (3) Plot Value Iteration and Policy Iteration:
#1. P and V iteration times: (Deter & Non-Deter)
a = pi_vi(P1, R1, 'more', 0, 4)[0]
b = pi_vi(P1, R1, 'more', 0, 4)[1]
print case_plot1(a, b)
a2 = pi_vi(P2, R2, 'more', 0, 4)[0]
b2 = pi_vi(P2, R2, 'more', 0, 4)[1]
print case_plot1(a2, b2)

#2. P and V values:
a = pi_vi(P1, R1, 'more', 0, 4)[4]
b = pi_vi(P1, R1, 'more', 0, 4)[5]
print case_plot1_value(a, b)
a2 = pi_vi(P2, R2, 'more', 0, 4)[4]
b2 = pi_vi(P2, R2, 'more', 0, 4)[5]
print case_plot1_value(a2, b2)

#3. P and V converge time:
at = pi_vi(P1, R1, 'more', 0, 4)[2]
bt = pi_vi(P1, R1, 'more', 0, 4)[3]
print plot_case_time(at, bt, 0, ql="without")
at2 = pi_vi(P2, R2, 'more', 0, 4)[2]
bt2 = pi_vi(P2, R2, 'more', 0, 4)[3]
print plot_case_time(at2, bt2, 0, ql="without")


# ---------------------------------------------------------------------
#### (2) Plot Q-Learning:
## 1> Values comparison with P, V:
base_value = QL_Strategy(P1, R1, "more", 0, 4)[0]
run_time = QL_Strategy(P1, R1, "more", 0, 4)[5]
values = QL_Strategy(P1, R1, "more", 0, 4)[4]
print case_plot2(values, run_time, base_value)

base_value = QL_Strategy(P2, R2, "more", 0, 4)[0]
run_time = QL_Strategy(P2, R2, "more", 0, 4)[5]
values = QL_Strategy(P2, R2, "more", 0, 4)[4]
print case_plot2(values, run_time, base_value)


## 2> Values and iter_times changes with greedy factors:
ql_t = plot_ql_time_gammas(P1, R1)
at = pi_vi(P1, R1, 'more', 0, 4)[2]
bt = pi_vi(P1, R1, 'more', 0, 4)[3]
print plot_case_time(at, bt, ql_t, "with")

ql_t = plot_ql_time_gammas(P2, R2)
at2 = pi_vi(P2, R2, 'more', 0, 4)[2]
bt2 = pi_vi(P2, R2, 'more', 0, 4)[3]
print plot_case_time(at2, bt2, ql_t, "with")


