import numpy as np
import scipy as sp
import sklearn
import math
from itertools import permutations

from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn import metrics

def kmeans_objective(data, means, clusters):
    k = len(means)
    return sum([np.sum(np.linalg.norm(data[np.where(clusters == i)] - means[i], axis=1)**2) for i in range(k)])

def dunn_index(data, labels, means):
    data_distances = metrics.pairwise.euclidean_distances(data)
    
    diam = [np.max(data_distances[np.where(labels==i)][:,np.where(labels==i)]) for i in range(k)]
    cluster_distances[cluster_distances == 0] = np.nan
    
    return np.min(np.nanmin(cluster_distances, axis=0))/np.max(diam)

def hubert_gamma(data, labels, means):
    n = data.shape[0]
    
    data_distances = metrics.pairwise.euclidean_distances(data)
    cluster_distances = metrics.pairwise.euclidean_distances(means)

    gamma_sum = 0
    for i in range(n):
        for j in range(i, n):
            gamma_sum += cluster_distances[labels[i],labels[j]] * data_distances[i,j] 
            
    return gamma_sum/(0.5*(n**2 - n))

def clustering_accuracy(pred_labels, actual_labels, k):
    accuracies = np.zeros(math.factorial(k))
    n = len(pred_labels)
    for ind1, i in enumerate(permutations(range(k))):
        for ind2, j in enumerate(range(k)):
            accuracies[ind1] += len(np.intersect1d(np.where(pred_labels == i[ind2]), np.where(actual_labels == j)))
    return np.max(accuracies)/n

def get_max_f1(pred_labels, actual_labels, k):
    f1_scores = np.zeros(math.factorial(k))
    
    def remap(entry, d):
        return d[entry]

    vremap = np.vectorize(remap)

    for i, perm in enumerate(permutations(range(k))):    
        d = dict(zip(range(k), perm))
        tmp_pred_lables = vremap(pred_labels, d)
        f1_scores[i] = metrics.f1_score(actual_labels, tmp_pred_lables, average='micro')
    
    return np.max(f1_scores)

## Furthest first traversal of dataset
def fft(data, k):
    i = 1
    
    centers = [np.random.choice(data.shape[0])]
    noncenter_data = data
    
    D = metrics.pairwise.euclidean_distances(data)
    
    while i < k: 
        dist=np.mean([D[j] for j in centers],0)
        for l in np.argsort(dist)[::-1]:
            if l not in centers:
                centers.append(l)
                i += 1
                break

    centers = np.array(centers)

    return data[centers]


def hartigan_algorithm(data, k):
    n = len(data)

    partitions = np.random.choice(k, size=(n))
    means = np.array([np.mean(data[np.where(partitions == i)], axis=0) for i in range(k)])

    sizes = np.bincount(partitions)
    assert len(sizes) == k

    i = 0

    while True:
        n_updates = 0
        for j in range(n):
            b = partitions[j]
            phi_old = sizes[b]/(sizes[b] - 1) * np.sum((means[b] - data[j])**2)
            costs = []
            for l in range(k):
                if l != b:
                    phi_new = sizes[l]/(sizes[l] + 1) * np.sum((means[l] - data[j])**2)
                    costs.append((phi_old - phi_new,  l))

            (cost_new, l) = max(costs)
                
            if cost_new >= 0:
                partitions[j] = l
                means[l] = (sizes[l]*means[l] + data[j])/(sizes[l]+1)
                means[b] = (sizes[b]*means[b] - data[j])/(sizes[b]-1)
                sizes[l] += 1
                sizes[b] -=1
                n_updates += 1
                
        i += 1
        print ('{}: {} changes'.format(i, n_updates))

        if n_updates == 0:
            break

    return means, partitions


##### DATA GOES HERE
## Number of clusters
K = 5
## Labels: Length N numpy array of ground truth labels
labels = []
## Data: N x D array of data 
data = np.zeros((N, d))

## Lloyd's with K-Means++ Initialization
## set n_init = 1, otherwise default is to run 10 Lloyd's/kmeans++ 10 times and pick the best one
kmeans_pp = cluster.KMeans(n_clusters=K, n_init=1).fit(data)
kmeans_objective(data, kmeans_pp.cluster_centers_, kmeans_pp.labels_)
clustering_accuracy(kmeans_pp.labels_, labels, K)

## Lloyd's with FFT Initialization
fft_init = fft(data, k)
kmeans_fft = cluster.KMeans(n_clusters=K, init=fft_init).fit(data)
kmeans_objective(data, kmeans_fft.cluster_centers_, kmeans_fft.labels_)
clustering_accuracy(kmeans_fft.labels_, labels, K)

## Lloyd's with Random Initialization
kmeans_rand = cluster.KMeans(n_clusters=10, init='random', n_init=1).fit(data)
kmeans_objective(data, kmeans_rand.cluster_centers_, kmeans_rand.labels_)
clustering_accuracy(kmeans_rand.labels_, labels)

## Hartigan's Method
hart_centers, hart_labels = hartigan_algorithm(data, K)
kmeans_objective(data, hart_centers, hart_labels)
clustering_accuracy(hart_labels, labels)


