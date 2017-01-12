from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn import mixture
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.lda import LDA
from sklearn.metrics import accuracy_score
import time
from sknn.mlp import Classifier, Layer

#########################################################################################

## Clustering Algorithms:
def cluster_process(process_name, n_c, X, y):
    tr_time=[]
    te_time=[]

    ## (1)K-means:
    if process_name == 'kmeans':  # oula distance
        kmean = KMeans(n_clusters=n_c, init='k-means++', n_init=20,
                       max_iter=500, tol=0.0001, precompute_distances = 'auto',
                       verbose=0, random_state=None, copy_x=True, n_jobs=1)

        start = time.time()
        kmean_result = kmean.fit(X)   # Compute k-means clustering
        new_data = kmean_result.transform(X)
        end = time.time()
        time_elapse1 = (end - start)
        tr_time.append(time_elapse1)

        start2 = time.time()
        y_predict = kmean_result.predict(X)  # Predict the closest cluster each sample in X belongs to
        end2 = time.time()
        time_elapse2 = (end2 - start2)
        te_time.append(time_elapse2)

        #### Performance valuation: Bounded range [-1, 1]:
        k_perform1 = metrics.v_measure_score(y, y_predict)
        k_perform2 = metrics.adjusted_mutual_info_score(y, y_predict)
        k_perform3 = metrics.adjusted_rand_score(y, y_predict)
        not_k_perform = metrics.silhouette_score(X, kmean_result.labels_)

        print '---------------------'
        print "K-means: selected number of clusters:", n_c
        print "Clustering performance given the ground truth class assignments:"
        print 'V measure ([0, 1]): ', k_perform1
        print 'Adjusted mutual_info measure ([0, 1]): ', k_perform2
        print 'Adjusted rand index ([-1, 1]): ', k_perform3
        print "Clustering performance without the ground truth class assignments: "
        print 'Silhouette coefficient ([-1, 1]): ', not_k_perform
        print "Train time:", tr_time
        print "Test time: ", te_time
        print "New clustered dataset:", new_data

        return k_perform1, k_perform2, k_perform3, not_k_perform, tr_time, te_time, new_data


    ## (2) EM (expectation maximization):
    elif process_name == 'em':
        em = mixture.GMM(n_components=n_c, n_init=1, n_iter=100, params='wmc',
            random_state=None, thresh=None, tol=0.001)

        start = time.time()
        em_result = em.fit(X)  # Compute k-means clustering
        new_data = em_result.sample(9298)
        end = time.time()
        time_elapse1 = (end - start)
        tr_time.append(time_elapse1)

        start2 = time.time()
        y_predict = em_result.predict(X) # Predict the closest cluster each sample in X belongs to
        end2 = time.time()
        time_elapse2 = (end2 - start2)
        te_time.append(time_elapse2)

        #### Performance valuation: Bounded range [-1, 1]:
        em_perform1 = metrics.v_measure_score(y, y_predict)
        em_perform2 = metrics.adjusted_mutual_info_score(y, y_predict)
        em_perform3 = metrics.adjusted_rand_score(y, y_predict)

        print '---------------------'
        print "EM: selected number of clusters:", n_c
        print "Clustering performance given the ground truth class assignments:"
        print 'V measure: ', em_perform1
        print 'Adjusted mutual_info measure: ', em_perform2
        print 'Adjusted rand index: ', em_perform3
        print "Train time:", tr_time
        print "Test time: ", te_time
        print "New clustered dataset:", new_data

        return em_perform1, em_perform2, em_perform3, tr_time, te_time, new_data






#########################################################################################
## Dimensionality Reduction Algorithms:

### (3)PCA:
def pca_process(n_comp, datasets):
    tr_time=[]
    pca = PCA(n_components = n_comp, whiten=True)                      # Components with maximum variance

    start = time.time()
    pca_result = pca.fit(datasets)
    reduced_data = pca_result.transform(datasets)
    end = time.time()
    time_elapse1 = (end - start)
    tr_time.append(time_elapse1)

    scores = pca_result.explained_variance_ratio_          # Percentage of variance explained for each components
    score = scores.sum()

    print "Selected number of components:", len(scores)
    print "Explained variance ratios: ", scores
    print "Total explained_variance_ratio: ", score
    print "Train time: ", tr_time
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    return reduced_data, scores, score, tr_time, len(scores)



### (4) ICA:
def ica_process(n_comp, datasets):
    ica = FastICA(n_components=n_comp, whiten=True, max_iter=100)
    tr_time=[]

    start = time.time()
    ica_result = ica.fit(datasets)
    reduced_data = ica_result.transform(datasets)
    end = time.time()
    time_elapse1 = (end - start)
    tr_time.append(time_elapse1)

    print 'ICA extract the top', n_comp, 'Independent components.'
    print "Train time: ", tr_time
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    return reduced_data, tr_time


### (5)Random Projection:
def rp_process(n_comp, datasets):
    tr_time1=[]
    tr_time2=[]
    rp_G = random_projection.GaussianRandomProjection(n_components=n_comp,eps=0.5, random_state=None)
    rp_S = random_projection.SparseRandomProjection(n_components=n_comp,eps=0.5, random_state=None)

# #Gaussian random matrix:
    start = time.time()
    rpG_result = rp_G.fit(datasets)
    reduced_data_G = rpG_result.transform(datasets)
    end = time.time()
    time_elapse1 = (end - start)
    tr_time1.append(time_elapse1)

# #Sparse random matrix:
    start2 = time.time()
    rpS_result = rp_S.fit(datasets)
    reduced_data_S = rpS_result.transform(datasets)
    end2 = time.time()
    time_elapse2 = (end2 - start2)
    tr_time2.append(time_elapse2)

    print "Number of components:", n_comp
    print "Train time for Gauissian and Sparse: ", tr_time1, tr_time2
    return reduced_data_G, reduced_data_S, tr_time1, tr_time2


### (6) LDA: it is a supervised method using known class labels
def lda_process(n_comp, datasets, y):
    tr_time=[]
    lda = LDA(n_components=n_comp,store_covariance=False, tol=0.0001, solver='svd', shrinkage=None, priors=None)  # Number of components (< n_classes - 1)
    start = time.time()
    lda_result = lda.fit(datasets, y)
    reduce_data = lda_result.transform(datasets)
    end = time.time()
    time_elapse = (end - start)
    tr_time.append(time_elapse)

    print "Number of components:", n_comp
    print "Train time: ", tr_time
    return reduce_data, tr_time








#########################################################################################
### (7) Neural Network:
def NN(XTrain, yTrain, XTest, yTestCorrect):
    Time_train=[]
    Time_test=[]
    tr_acc=[]
    te_acc=[]

    clf = Classifier(layers=[Layer("Sigmoid", units=13),
                             Layer("Sigmoid", units=13),
                             Layer("Sigmoid")],learning_rate=0.1, n_iter=30)

    start = time.time()
    clf = clf.fit(XTrain, yTrain)
    end = time.time()
    time_elapse1 = (end - start)
    Time_train.append(time_elapse1)

    Train_predicted = clf.predict(XTrain)
    Train_accs = accuracy_score(yTrain, Train_predicted) #Training accuracy
    tr_acc.append(Train_accs)

    start2 = time.time()
    Test_predicted = clf.predict(XTest)
    end2 = time.time()
    time_elapse2 = (end2 - start2)
    Time_test.append(time_elapse2)

    Test_accs = accuracy_score(yTestCorrect, Test_predicted) #Testing accuracy.
    te_acc.append(Test_accs)
    print "Train and Test accuracies: ", tr_acc, te_acc
    print "Train and Test time: ", Time_train, Time_test
    return tr_acc, te_acc, Time_train, Time_test

