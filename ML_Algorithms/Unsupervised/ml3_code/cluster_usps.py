import Create_Dataset
from ULA import *
import matplotlib.pyplot as plt


USPS = Create_Dataset.create_dataset('/Users/yywxenia/PycharmProjects/MLProj3/USPS_Digit_Data.txt',0,256,256)
###(2) Training and Testing data
n_samples = len(USPS['target'])
print "Number of samples: ", n_samples
X = USPS['data']
y = USPS['target']

kmean1 = []
em1 = []
kmean2 = []
em2 = []
kmean3 = []
em3 = []
kmean4 = []

trainK=[]
testK=[]
trainE=[]
testE=[]

cluster=np.int32(np.linspace(3, 20, 10))
for i in cluster:
    a = cluster_process('kmeans', i, X, y)
    b= cluster_process('em', i, X, y)
    kmean1.append(a[0])
    em1.append(b[0])
    kmean2.append(a[1])
    em2.append(b[1])
    kmean3.append(a[2])
    em3.append(b[2])
    kmean4.append(a[3])

    trainK.append(a[4])
    testK.append(a[5])
    trainE.append(b[3])
    testE.append(b[4])



## Ploting the performances:
i_axis=cluster

plt.plot(i_axis, kmean1,"r.-",i_axis, em1,'b.-',linewidth=1.5)
plt.legend(['K-means', 'EM'], loc='lower right')
plt.ylabel('Clustering performance')
plt.xlabel('Cluster number')
plt.title("V measure: Performance of K-means and EM")
plt.show()

plt.plot(i_axis, kmean2,"r.-",i_axis, em2,'b.-',linewidth=1.5)
plt.legend(['K-means', 'EM'], loc='lower right')
plt.ylabel('Clustering performance')
plt.xlabel('Cluster number')
plt.title("Adjusted mutual_info: Performance of K-means and EM")
plt.show()

plt.plot(i_axis, kmean3,"r.-",i_axis, em3,'b.-',linewidth=1.5)
plt.legend(['K-means', 'EM'], loc='lower right')
plt.ylabel('Clustering performance')
plt.xlabel('Cluster number')
plt.title("Adjusted rand index: Performance of K-means and EM")
plt.show()

plt.plot(i_axis, kmean4,"r.-",linewidth=1.5)
plt.legend(['K-means'], loc='lower right')
plt.ylabel('Clustering performance')
plt.xlabel('Cluster number')
plt.title("Silhouette coefficient: Performance of K-means")
plt.show()


#### for train and test time:
## kmeans
plt.subplot(211)
plt.plot(i_axis, trainK, "c", label='Train time',linewidth=1.5)
plt.legend()
plt.ylabel('Time')
plt.xlabel('Cluster number')
plt.title("Train and Test Time of K-means")

plt.subplot(212)
plt.plot(i_axis, testK, 'm', label='Test time',linewidth=1.5)
plt.ylabel('Time')
plt.xlabel('Cluster number')
plt.legend()
plt.show()


## EM
plt.subplot(211)
plt.plot(i_axis, trainE, "c", label='Train time',linewidth=1.5)
plt.legend()
plt.ylabel('Time')
plt.xlabel('Cluster number')
plt.title("Train and Test Time of EM")

plt.subplot(212)
plt.plot(i_axis, testE, 'm', label='Test time',linewidth=1.5)
plt.ylabel('Time')
plt.xlabel('Cluster number')
plt.legend()
plt.show()