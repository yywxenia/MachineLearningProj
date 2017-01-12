import Create_Dataset
from ULA import *
import matplotlib.pyplot as plt

iris = Create_Dataset.create_dataset('/Users/yywxenia/PycharmProjects/MLProj3/iris.data.txt',0,4,4)
###(2) Training and Testing data

X = iris['data']
y = iris['target']

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

i_number=[6,5,4,3,2]
for i in i_number:
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
i_axis=[6,5,4,3,2]

plt.plot(i_axis, kmean1,"r.-",i_axis, em1,'b.-',linewidth=1.5)
plt.legend(['K-means', 'EM'], loc='upper right')
plt.ylabel('Clustering performance')
plt.xlabel('Cluster number')
plt.title("V measure: Performance of K-means and EM")
plt.show()

plt.plot(i_axis, kmean2,"r.-",i_axis, em2,'b.-',linewidth=1.5)
plt.legend(['K-means', 'EM'], loc='upper right')
plt.ylabel('Clustering performance')
plt.xlabel('Cluster number')
plt.title("Adjusted mutual_info: Performance of K-means and EM")
plt.show()

plt.plot(i_axis, kmean3,"r.-",i_axis, em3,'b.-',linewidth=1.5)
plt.legend(['K-means', 'EM'], loc='upper right')
plt.ylabel('Clustering performance')
plt.xlabel('Cluster number')
plt.title("Adjusted rand index: Performance of K-means and EM")
plt.show()

plt.plot(i_axis, kmean4,"r.-",linewidth=1.5)
plt.legend(['K-means'], loc='upper right')
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