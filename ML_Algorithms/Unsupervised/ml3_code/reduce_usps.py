
import Create_Dataset
from ULA import *
import matplotlib.pyplot as plt
import numpy as np

USPS = Create_Dataset.create_dataset('/Users/yywxenia/PycharmProjects/MLProj3/USPS_Digit_Data.txt',0,256,256)
X = USPS['data']
y = USPS['target']


########################################################################################
##  Implement PCA, ICA, RP, LDA algorithms on USPS datasets:
########################################################################################

## (1) PCA:
print '(1) Apply PCA reduction on USPS digit data:'
print '\n'
pca_process(256, X)      # 92.5% of the data can be described by the first components.
print "Data shape:", X.shape

print "------------------------------------------------------------"
## Select components by changing total explained variance ratio:
score_list=[]
time_list=[]
num_c=[]

pent=[0.1,  0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9, 0.95]
for c_percent in pent:
    result = pca_process(c_percent, X)
    num_c.append(result[4])
    score_list.append(result[2])
    time_list.append(result[3])

print "Scores: ", score_list
print "Train time: ", time_list
print "Number of components selected: ", num_c


plt.subplot(311)
plt.plot(pent, num_c,"r.-", label='Selected components')
plt.legend(loc='lower right')
plt.ylabel('Components')
plt.xlabel('Pre-set total variance explained ratio')
plt.title("PCA: select components for USPS data")

plt.subplot(312)
plt.plot(num_c, score_list, 'b.-', label='Real total explained variance ratio')
plt.ylabel('Results')
plt.xlabel('Number of components')
plt.legend(loc='lower right')

plt.subplot(313)
plt.plot(num_c, time_list, 'b.-', label='Training time')
plt.ylabel('Train time')
plt.xlabel('Number of components')
plt.legend(loc='lower right')
plt.show()

print "===========================Finished PCA==============================="
print '(2) Apply ICA reduction on USPS digit data:'
print '\n'
pca_process(256, X)      # 92.5% of the data can be described by the first components.
print "Data shape:", X.shape
print "------------------------------------------------------------"

time_list=[]
num = np.int32(np.linspace(3, 256, 10))
for c in num:
    result = ica_process(c, X)
    time_list.append(result[1])
print "Train time: ", time_list
print "Number of components selected: ", num
plt.plot(num,time_list, "r.-", label='Selected components')
plt.legend(loc='lower right')
plt.ylabel('Train time')
plt.xlabel('Components')
plt.title("ICA: USPS data")


### Plot the original and reduced data:
y_example = y[0:9] #y represent the labels, so label 0 means number "1", label 1 means number "2", etc.
print y_example
for i in range(0, 9):
    a=X[i]
    idx="33"+ str(i+1)
    plt.subplot(idx)
    a = np.asarray(a)
    a = a.reshape(16,16)
    plt.imshow(a)

X_r=ica_process(36, X)[0]

for i in range(0, 9):
    a=X_r[i]
    idx="33"+ str(i+1)
    plt.subplot(idx)
    a = np.asarray(a)
    a = a.reshape(6, 6)
    plt.imshow(a)

plt.show()
print "===========================Finished ICA==============================="

components = np.int32(np.linspace(3, 256, 10))
G_time=[]
S_time=[]
for n_comp in components:
    # create the random projection
    a = rp_process(n_comp, X)
    G_time.append(a[2])
    S_time.append(a[3])
    print '~~~~~~~~~~~~~~~~~~~~~~~'

plt.plot(components, G_time,"c.-", components, S_time,'m.-')
plt.legend(['Gauissian RP', 'Sparse RP'], loc='lower right')
plt.ylabel('Time')
plt.xlabel('Components')
plt.title("Time change by increasing number of components")
plt.show()

print "=====================Finished Random Projection======================="
num = np.int32(np.linspace(3, 256, 10))
time_list=[]
for c in num:
    result = lda_process(c, X, y)
    time_list.append(result[1])
    print '~~~~~~~~~~~~~~~~~'

plt.plot(num, time_list, "m.-", label='Time')
plt.legend(loc='lower right')
plt.xlabel('Components')
plt.ylabel('Train time')
plt.title("LDA: usps data")
plt.show()

print "===========================Finished LDA==============================="

