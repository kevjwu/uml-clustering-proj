import numpy as np
import scipy as sp
import sklearn
from itertools import combinations, permutations
import math

from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn import metrics

class GaussianClusterGenerator(object):
  
    def __init__(self, d, k, n, mean_limits, wishart_param):
        self.d = d
        self.k = k
        self.n = n
        self.wishart_param = wishart_param

        self.means = np.array([np.random.uniform(low=mean_limits[0], high=mean_limits[1], size=(d,)) for i in range(k)])
        self.variances = [sp.stats.invwishart.rvs(df=d, scale=np.eye(d)*self.wishart_param) for i in range(k)]

        self.data = []
        self.labels = []
        
        ## Higher --> more separable clusters
        avg_intercluster_dist = np.mean(([np.sum((self.means[i] - self.means[j])**2)**0.5 for i, j in combinations(range(self.k), 2)]))
        avg_spread = np.mean([np.sum(np.diagonal(self.variances[i]))**0.5 for i in range(self.k)])
        self.separation_ratio = avg_intercluster_dist / avg_spread 

    def generate_dataset(self):
    
        ## N x d dataset of cluster centers
    
        for i in range(self.n):
            j = np.random.choice(self.k)
            self.data.append(np.random.multivariate_normal(self.means[j], self.variances[j]))
            self.labels.append(j) 

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
    
        return self.data, self.labels
      
    def visualize(self, centers=None):
        if centers is None:
            centers = self.means
            
        if self.d == 2:
            data_2d = self.data
            centers_2d = centers
        else:
            data_2d = sklearn.manifold.TSNE(n_components=2).fit_transform(np.concatenate((datagen.data, datagen.means)))
            centers_2d = data_2d[-len(centers):]
            data_2d = data_2d[:-len(centers)]

        plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=self.labels, cmap='nipy_spectral')
        plt.scatter(x=centers_2d[:,0], y=centers_2d[:,1], c='black', marker='+')
        return
