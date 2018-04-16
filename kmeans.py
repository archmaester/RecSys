

import pandas as pd
import numpy as np
import numpy.linalg as la
# import matplotlib.pyplot as plt
# import matplotlib.colors as clr
import sklearn.metrics as m
import sklearn.metrics.pairwise as skm

def cosine(X, Y):
    t1 = np.reshape(X,(-1,1))
    t2 = np.reshape(Y,(-1,1))
    return np.dot(t1.T,t2)
def kmeans(X, k):
    '''
        X: dataset
        k: no of clusters
        lab: correct labels
        
        Returns: predicted labels by k-means algorithm
    '''        
    n, d = X.shape
    ind = np.arange(n)
    np.random.shuffle(ind)
    k_centres = X[ind[0:k]]
    for i in range(k):
        k_centres[i] = k_centres[i]/np.linalg.norm(k_centres[i])
    Clusters = np.zeros(n)
    cluster_changes = True
    while cluster_changes:
        cluster_changes = False
        for i in range(n):
            c = Clusters[i]
            Clusters[i] = np.argmax([cosine(X[i], k_centres[j]) for j in range(k)])
                                                           # Updating cluster of this point
            if c != Clusters[i]:
                cluster_changes = True
        for i in range(k):                                      # updating cluster centers
            k_centres[i] = np.sum([X[j] for j in range(n) if Clusters[j] == i ], axis = 0)
            k_centres[i] = k_centres[i]/np.linalg.norm(k_centres[i])
            
    return(Clusters, k_centres)

