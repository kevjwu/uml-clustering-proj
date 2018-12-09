import random
import numpy as np
import math
import kmeans as kevin
import operator
import sklearn.metrics as metrics
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import cluster
import collections
import numpy.linalg as LA
from scipy.stats import logistic


def sample_from_ecllipse(a_data, center_data, error):
	X = []
	for i in xrange(0, len(a_data)):
		# X.append(np.random.normal(0, a_data[i] * a_data[i]))
		X.append(a_data[i] * 2 * (np.random.random() - 0.5))
	X = np.array(X)
	d = math.sqrt(sum(np.square(X) / np.square(a_data)))
	error_list = np.random.normal(0, error, len(X))
	return (X / d + center_data) + error_list

def generate_ellipse_data(a_data_list, center_list, n_samples, error = 0.0):
	X = []
	y = []
	for i in xrange(0, n_samples):
		current_cluster = random.randint(0, len(center_list) - 1)
		X.append(sample_from_ecllipse(a_data_list[current_cluster], center_list[current_cluster], error))
		y.append(current_cluster)
	return np.array(X),np.array(y)

fig, ax = plt.subplots(4, 4, figsize = (100, 100))
fig.subplots_adjust(hspace=.5)


def plot(i, j, X, label, title):
	reds = label == 0
	blues = label == 1
	green = label == 2
	black = label == 3
	yellow = label == 4

	ax[i][j].scatter(X[reds, 0], X[reds, 1], c = "red", s = 20, edgecolor = 'k')
	ax[i][j].scatter(X[blues, 0], X[blues, 1], c = "blue",s = 20, edgecolor = 'k')
	ax[i][j].scatter(X[green, 0], X[green, 1], c = "green",s = 20, edgecolor = 'k')
	ax[i][j].scatter(X[black, 0], X[black, 1], c = "black",s = 20, edgecolor = 'k')
	ax[i][j].scatter(X[yellow, 0], X[yellow, 1], c = "yellow",s = 20, edgecolor = 'k')
	ax[i][j].set_title(title)
	ax[i][j].autoscale(False)

	return

a_data_list = np.array([
	[1.0,3.0],
	[2.0,1.0],
	[1.0,1.0]
])

center_list = np.array([
	[0.0,0.0],
	[0.1,0.0],
	[0.0,1.0]
])


def get_ellipse_params(K, d, s):
	# this generates centers and params depending on inverse_separability ([0,1]),
	# if s is 0 then it gives ellipses same center and each having strictly greater dimension
	# the more we increase s, theri centers get separated  

	a_data_list = np.random.rand(K,d) + 0.01
	center_list = np.random.rand(K,d)

	center_list = np.zeros((K, d))
	for i in xrange(1, K):
		# a_data_list[i] = 1.0 / (1 + s) * a_data_list[i - 1] + (1 - s) * a_data_list[i]
		# center_list[i] = 1.0 / (1 + s) * center_list[i - 1] + (1 - s) * center_list[i]

		a_data_list[i] = (1 - s)* a_data_list[i - 1] + (s) * a_data_list[i]
		# center_list[i] = (1 - s) * center_list[i - 1] + (s/4) * center_list[i]
		# center_list[i] = (1 - s) * center_list[i - 1] + (s) * center_list[i]



	return a_data_list, center_list

d = 2
K = 3
s = 0




def relabel_data(y_groud_truth, y_clustering, K):
	unassigned_gt = set([i for i in xrange(0, K)])
	unassigned_cl = set([i for i in xrange(0, K)])
	best_data = []
	match_matrix = [[0 for j in xrange(0, K)] for i in xrange(0, K)]


	for (gt_index, cl_index) in zip(y_groud_truth, y_clustering):
			match_matrix[gt_index][cl_index] += 1

	match_data = []
	for i in xrange(0, K):
		for j in xrange(0, K):
			match_data.append((i,j,match_matrix[i][j]))

	mapping = {}
	for i in xrange(0, K):
		(gt_index, cl_index, match_count) = max(match_data, key = lambda x: x[2])
		match_data = filter(lambda x: x[0] != gt_index and x[1] != cl_index, match_data)
		mapping[cl_index] = gt_index

	# print mapping

	Z = [label for label in y_clustering]

	for i in xrange(0, len(Z)):
		Z[i] = mapping[y_clustering[i]]

	return Z


# ###### center stretching ##########
# a_data_list = np.random.rand(K, d) + 0.01
# center_list = np.random.rand(K,d)
# l0 = np.max(a_data_list) / 8

# center_list = center_list - np.mean(center_list)
# for i in xrange(0, 8):
# 	l = l0 * 10 * float(i)
# 	center_list_i = center_list * l
# 	X,y = generate_ellipse_data(a_data_list, center_list_i, 1000, 0.01)
# 	kmeans_pp = cluster.KMeans(n_clusters = K, n_init = 1).fit(X)
# 	y_kmeans_pp = kmeans_pp.labels_
# 	y_pred = relabel_data(y, y_kmeans_pp, K)
# 	f1_score = metrics.f1_score(y, y_pred, average = 'weighted')
# 	sil_score = metrics.silhouette_score(X,y)
# 	print f1_score, sil_score
# 	plot(i / 4, i % 4,X, y, "Original")












# ##### weird generation #############
# s_list = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90 ,1.0]
# a_data_list = np.random.rand(K,d) + 0.01

# for i in xrange(1, K):
# 	a_data_list[i] = a_data_list[i] / 5.0  + a_data_list[i - 1] + 0.1

# center_list = np.zeros((K, d))
# for i in xrange(0, 8):
# 	# a_data_list, center_list = get_ellipse_params(K, d, s_list[i])
# 	X,y = generate_ellipse_data(a_data_list, center_list, 1000, (1.0 - s_list[i] * 0.9)/10.0)

# 	kmeans_pp = cluster.KMeans(n_clusters = K, n_init = 1).fit(X)
# 	y_kmeans_pp = kmeans_pp.labels_
# 	y_pred = relabel_data(y, y_kmeans_pp, K)
# 	f1_score = metrics.f1_score(y, y_pred, average = 'weighted')
# 	sil_score = metrics.silhouette_score(X,y)
# 	print f1_score, sil_score
# 	plot(i / 4, i % 4,X, y, "Original")


def dbscan(X):
	dbscan = cluster.DBSCAN(eps = 0.1).fit(X)
	y_dbscan = dbscan.labels_

	z =  collections.Counter(y_dbscan)
	top_data = z.most_common(K)

	top_mapping = {}
	j = 0
	for top_cluster_index, value in top_data:
		top_mapping[top_cluster_index] = j
		j += 1

	best_clusters = set(map(lambda x : x[0], top_data))
	for j in xrange(0, len(y_dbscan)):
		if y_dbscan[j] not in best_clusters:
			y_dbscan[j] = -1
		else:
			y_dbscan[j] = top_mapping[y_dbscan[j]]

	valid_clusters = set([j for j in xrange(0, K)])
	y_valid = [1.0 for j in xrange(0, len(y_dbscan))]
	for j in xrange(0, len(y_dbscan)):
		if y_dbscan[j] not in  valid_clusters:
			y_valid[j] = float('inf')
	y_valid = np.array(y_valid)

	for j in xrange(0, len(y_dbscan)):
		if y_dbscan[j] not in  valid_clusters:
			D_list = LA.norm(X - X[j]) * y_valid
			y_dbscan[j] = y_dbscan[np.argmin(D_list)]

	return np.array(y_dbscan)

##### weird generation #############
# s_list = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90 ,1.0]
# s_list = [0.4, 0.45, 0.50, 0.55, 0.60, 0.75, 0.90 ,1.0]
e = 8

s_list = [logistic.cdf( 0 + 3 * float(i) / float(e)) for i in xrange(0, e)]
print s_list
a_data_list = np.random.rand(K,d) + 0.01

for i in xrange(1, K):
	a_data_list[i] = a_data_list[i] / 5.0  + a_data_list[i - 1] + 0.1

center_list = np.zeros((K, d))


# def get_nearest_cluster(X, y_aggl, point, y_valid):
	
# 	return y_aggl[np.argmin(D_list)]


for i in xrange(0, 8):
	# print 'here'
	# a_data_list, center_list = get_ellipse_params(K, d, s_list[i])
	X,y = generate_ellipse_data(a_data_list, center_list, 1000, (1.0 - s_list[i])/10.0)
	
	# y_aggl = dbscan(X)
	# y_aggl = cluster.SpectralClustering(n_clusters = K, n_jobs = -1).fit(X).labels_
	y_aggl = cluster.AgglomerativeClustering(n_clusters = K, linkage = 'single').fit(X).labels_


	y_pred = np.array(relabel_data(y, y_aggl, K))

	# print set(y_pred)
	# print collections.Counter(y_pred)
	# print set(y)
	# print y_pred
	f1_score = metrics.f1_score(y, y_pred, average = 'weighted')
	sil_score = metrics.silhouette_score(X,y)
	print f1_score, sil_score
	plot(2 * (i / 4), i % 4, X, y, "Original")
	plot(2 * (i / 4) + 1, i % 4, X, y_pred, "DBSCAN")




plt.show()


# X_embedded = TSNE(n_components = 2).fit_transform(X)
# plot(0,1,X_embedded, y, "TSNE")
quit()

a_data_list = np.array([
	[1.0,3.0,3,5],
	[5,2.0,1.0,1],
	[1.0,5,1.0,5]
])

center_list = np.array([
	[0.0,0.0,0.0,0.0],
	[1.0,1.0,3.4,1.0],
	[-1.0,0.0,0.0,-1.0]
])

X,y = generate_ellipse_data(a_data_list, center_list, 100)


# X_embedded = TSNE(n_components = 2).fit_transform(X)

# plot(0,0,X, y, "2d ellipse")
# plot(1,0,X_embedded, y, "TSNE")





# experiments on dimensions :
# for d in [2,3,4,5,6,7,8,9,10]:
    



# experiments on separability :





# K = 3



# kmeans_pp = cluster.KMeans(n_clusters = K, n_init = 1).fit(X)
# y_kmeans_pp = kmeans_pp.labels_

# y_pred = relabel_data(y, y_kmeans_pp, K)
# y_pred = remap(y, y_kmeans_pp,K)

# f1_score = metrics.f1_score(y, y_pred, average = 'weighted')

# print f1_score


# kmeans_objective(data, kmeans_pp.cluster_centers_, kmeans_pp.labels_)
# clustering_accuracy(kmeans_pp.labels_, labels, K)

## Lloyd's with FFT Initialization
# fft_init = kevin.fft(X, K)
# kmeans_fft = cluster.KMeans(n_clusters = K, init = fft_init).fit(X)
# y_keans_fft = kmeans_fft.labels_

# dbscan = cluster.DBSCAN().fit(X)
# y_dbscan = dbscan.labels_
# print y_dbscan
# plot(1,1,X_embedded, y_dbscan, "DBSCAN")
# plot(1,2,X_embedded, y_kmeans_pp, "KMEANS")


# plt.autoscale(False)
plt.show()