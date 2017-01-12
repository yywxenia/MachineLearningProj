###(1) Import data
import Create_Dataset
import numpy as np
import pylab as plt

USPS = Create_Dataset.create_dataset('/Users/yywxenia/Desktop/USPS_Digit_Data.txt',0,256,256)


###(2) Visulize 9 examples data of USPS
X = USPS['data']
print len(X[0])
y = USPS['target']

y_example = y[0:9] #y represent the labels, so label 0 means number "1", label 1 means number "2", etc.
print y_example

for i in range(0,9):
    a=X[i]
    idx="33"+ str(i+1)
    plt.subplot(idx)
    a = np.asarray(a)
    a=a.reshape((16,16))

    plt.imshow(a)
plt.show()
