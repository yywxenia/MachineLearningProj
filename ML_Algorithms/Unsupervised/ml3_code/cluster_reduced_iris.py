import Create_Dataset
from ULA import *
import matplotlib.pyplot as plt


iris = Create_Dataset.create_dataset('/Users/yywxenia/PycharmProjects/MLProj3/iris.data.txt',0,4,4)
X = iris['data']
y = iris['target']

#########################################################################################
### Run Cluster on dimensionality-reduction USPS data:
#########################################################################################
components=[2, 3, 4]
print "Number of selected components:", components
clusters= [2,3,4,5,6]
print "Number of selected clusters:", clusters

print "(1) Run Kmeans on dimension-reduction iris data:"
for n_comp in components:
    print '============================================================='
    print 'Number of components for PCA reduction iris data: ', n_comp
    kmeanA=[]
    kmeanB=[]
    kmeanC=[]
    kmeanD=[]
    kmeanE=[]

    for i in clusters:
        print '---------------------'
        print "selected number of clusters: ", i

        X_pca = pca_process(n_comp, X)[0]
        X_ica = pca_process(n_comp, X)[0]
        X_rpg = rp_process(n_comp, X)[0]
        X_rps = rp_process(n_comp, X)[1]
        X_lda = lda_process(n_comp, X, y)[0]

        a = cluster_process('kmeans', i, X_pca, y)
        b = cluster_process('kmeans', i, X_ica, y)
        c = cluster_process('kmeans', i, X_rpg, y)
        d = cluster_process('kmeans', i, X_rps, y)
        e = cluster_process('kmeans', i, X_lda, y)

        kmeanA.append(a[1])
        kmeanB.append(b[1])
        kmeanC.append(c[1])
        kmeanD.append(d[1])
        kmeanE.append(e[1])

    plt.plot(clusters, kmeanA,linewidth=3, color = "r", label='PCA')
    plt.plot(clusters, kmeanB, linewidth=2, color = "b", label='ICA')
    plt.plot(clusters, kmeanC, linewidth=2, color = "c", label='RPG')
    plt.plot(clusters, kmeanD, linewidth=2, color = "m", label='RPS')
    plt.plot(clusters, kmeanE, linewidth=2, color = "k", label='LDA')
    plt.legend(loc='upper left')
    plt.ylabel('Performance scores')
    plt.xlabel('K-means Clusters')


print "(1) Run EM on dimension-reduction iris data:"
for n_comp in components:
    print '============================================================='
    print 'Number of components for PCA reduction iris data: ', n_comp
    emA=[]
    emB=[]
    emC=[]
    emD=[]
    emE=[]

    for i in clusters:
        print '---------------------'
        print "selected number of clusters: ", i

        X_pca = pca_process(n_comp, X)[0]
        X_ica = pca_process(n_comp, X)[0]
        X_rpg = rp_process(n_comp, X)[0]
        X_rps = rp_process(n_comp, X)[1]
        X_lda = lda_process(n_comp, X, y)[0]

        a = cluster_process('em', i, X_pca, y)
        b = cluster_process('em', i, X_ica, y)
        c = cluster_process('em', i, X_rpg, y)
        d = cluster_process('em', i, X_rps, y)
        e = cluster_process('em', i, X_lda, y)

        emA.append(a[1])
        emB.append(b[1])
        emC.append(c[1])
        emD.append(d[1])
        emE.append(e[1])

    plt.plot(clusters, emA,linewidth=2, color = "r", label='PCA')
    plt.plot(clusters, emB, linewidth=2, color = "b", label='ICA')
    plt.plot(clusters, emC, linewidth=2, color = "c", label='RPG')
    plt.plot(clusters, emD, linewidth=2, color = "m", label='RPS')
    plt.plot(clusters, emE, linewidth=2, color = "k", label='LDA')
    plt.legend(loc='upper left')
    plt.ylabel('Performance scores')
    plt.xlabel('EM Clusters')
    plt.show()
print "===========================THE END================================="
