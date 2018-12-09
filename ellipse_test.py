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
from sklearn import mixture
import collections
import numpy.linalg as LA
from scipy.stats import logistic
import statsmodels.api as sm


def sample_from_ecllipse(a_data, center_data, error):
	X = []
	for i in xrange(0, len(a_data)):
		X.append(np.random.normal(0, a_data[i] * a_data[i]))
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

	Z = [label for label in y_clustering]

	for i in xrange(0, len(Z)):
		Z[i] = mapping[y_clustering[i]]

	return Z

###### center stretching ##########

def get_cluster_centers(X, y, K):
	C = np.random.rand(K, len(X[0]))

	for i in xrange(0, K):
		cluster_bit_vector = np.zeros(len(X))
		for label in y:
			if label == i:
				cluster_bit_vector[i] = 1.0
		C[i] = np.mean( cluster_bit_vector * X )

	return C


def get_kmeans_firthest(X, y, K, max_trial):
	f1_list = []
	for i in xrange(0, max_trial):
		kmeans_pp = cluster.KMeans(n_clusters = K, init = kevin.fft(X, K), n_jobs = -1).fit(X)
		y_kmeans_pp = kmeans_pp.labels_
		y_pred = relabel_data(y, y_kmeans_pp, K)
		f1_score = metrics.f1_score(y, y_pred, average = 'weighted')
		f1_list.append(f1_score)
	# f1_list = np.array(f1_list)
	return f1_list
	# return np.mean(f1_list), np.min(f1_list), np.max(f1_list), np.var(f1_list)


def get_kmeans_plus(X, y, K, max_trial):
	f1_list = []
	for i in xrange(0, max_trial):
		kmeans_pp = cluster.KMeans(n_clusters = K, n_init = 1, n_jobs = -1).fit(X)
		y_kmeans_pp = kmeans_pp.labels_
		y_pred = relabel_data(y, y_kmeans_pp, K)
		f1_score = metrics.f1_score(y, y_pred, average = 'weighted')
		f1_list.append(f1_score)
	# f1_list = np.array(f1_list)
	return f1_list
	# return np.mean(f1_list), np.min(f1_list), np.max(f1_list), np.var(f1_list)

def get_gaussian(X, y, K, max_trial):
	f1_list = []
	for i in xrange(0, max_trial):
		y_gmc = mixture.GaussianMixture(n_components = K).fit_predict(X)
		y_pred = relabel_data(y, y_gmc, K)
		f1_score = metrics.f1_score(y, y_pred, average = 'weighted')
		f1_list.append(f1_score)
	# f1_list = np.array(f1_list)
	return f1_list
	# return np.mean(f1_list), np.min(f1_list), np.max(f1_list), np.var(f1_list)

# def get_hartigan(X, y, K, max_trial):
# 	f1_list = []
# 	for i in xrange(0, max_trial):
# 		y_gmc = mixture.GaussianMixture(n_components = K).fit_predict(X)
# 		y_pred = relabel_data(y, y_gmc, K)
# 		f1_score = metrics.f1_score(y, y_pred, average = 'weighted')
# 		f1_list.append(f1_score)
# 	f1_list = np.array(f1_list)
# 	return np.mean(f1_list), np.min(f1_list), np.max(f1_list), np.var(f1_list)

def get_agglomerative(X, y, K, max_trial):
	f1_list = []
	for i in xrange(0, max_trial):
		aggl = cluster.AgglomerativeClustering(n_clusters = K, linkage = 'single').fit(X)
		y_aggl = aggl.labels_
		y_pred = relabel_data(y, y_aggl, K)
		f1_score = metrics.f1_score(y, y_pred, average = 'weighted')
		f1_list.append(f1_score)
	# f1_list = np.array(f1_list)
	return f1_list
	# return np.mean(f1_list), np.min(f1_list), np.max(f1_list), np.var(f1_list)


def dbscan(X):
	dbscan = cluster.DBSCAN(eps = 0.1, n_jobs = -1).fit(X)
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

def get_dbscan(X, y, K, max_trial):
	f1_list = []
	for i in xrange(0, max_trial):
		y_dbscan = dbscan(X)
		y_pred = relabel_data(y, y_dbscan, K)
		f1_score = metrics.f1_score(y, y_pred, average = 'weighted')
		f1_list.append(f1_score)
	# f1_list = np.array(f1_list)
	return f1_list
	# return np.mean(f1_list), np.min(f1_list), np.max(f1_list), np.var(f1_list)


def get_spectral(X, y, K, max_trial):
	f1_list = []
	for i in xrange(0, max_trial):
		spec = cluster.SpectralClustering(n_clusters = K, n_jobs = -1).fit(X)
		y_spec = spec.labels_
		y_pred = relabel_data(y, y_spec, K)
		f1_score = metrics.f1_score(y, y_pred, average = 'weighted')
		f1_list.append(f1_score)
	# f1_list = f1_list
	# return np.mean(f1_list), np.min(f1_list), np.max(f1_list), np.var(f1_list)
	return f1_list

max_trial = 2
d_list = [2, 4 ,5, 6, 10]
K = 5
N = 1000
s_trial = 30


algo_list = [ get_kmeans_plus, get_gaussian, get_agglomerative, get_dbscan, get_kmeans_firthest, get_spectral]
algo_name_list = [ "KM++", "GMM", "AGL", "DBSCAN", "KM-FF", "SPECTRAL"]

d_f1 = [[] for i in xrange(0,len(algo_list))]


for d_index in xrange(0, len(d_list)):
	d = d_list[d_index]

	f1_list = [[] for i in xrange(0,len(algo_list))]
	sil_list = [[] for i in xrange(0,len(algo_list))]	
	s_list = [logistic.cdf( 0 + 3 * float(i) / float(s_trial)) for i in xrange(0, s_trial)]

	print s_list

	a_data_list = np.random.rand(K,d) + 0.01
	for i in xrange(1, K):
		a_data_list[i] = a_data_list[i] / 5.0  + a_data_list[i - 1] + 0.1
	center_list = np.zeros((K, d))

	for s_index in xrange(0, len(s_list)):
		print "D = ",d,"S = ",s_index,"s_val = ", s_list[s_index]
		X,y = generate_ellipse_data(a_data_list, center_list, N, (1.0 - s_list[s_index])/10.0)

		#### RUN EACH OF THE ALGO
		for algo_index in xrange(0, len(algo_list)):
			algo = algo_list[algo_index]
			f1_values = algo(X, y, K, max_trial)
			f1_list[algo_index] += [np.mean(f1_values)]
			# sil_list[algo_index] += [s_list[s_index] for r in xrange(0, len(f1_values))]
			sil_list[algo_index] += [s_list[s_index]]

	for algo_index in xrange(0, len(algo_list)):
		# pf = sm.nonparametric.lowess(np.array(f1_list[algo_index]), np.array(sil_list[algo_index]), frac = 0.4)
		# new_s = pf[:,0]
		# new_mean = pf[:,1]

		x = sil_list[algo_index]
		y = f1_list[algo_index]


		plt.plot(x, y, label = algo_name_list[algo_index])

	plt.legend()
	plt.xlabel("separability", fontsize = 20)
	plt.ylabel("weighted-f1", fontsize = 20)
	plt.savefig(str(d) + "_performance.png", dpi = 300, bbox_inches = 'tight')
	plt.clf()
	plt.cla()


	for algo_index in xrange(0, len(algo_list)):
		d_f1[algo_index].append(f1_list[algo_index][-1])


for algo_index in xrange(0, len(algo_list)):
	x = d_list
	y = d_f1[algo_index]
	plt.plot(x, y, label = algo_name_list[algo_index])

plt.legend()
plt.xlabel("dimension", fontsize = 20)
plt.ylabel("weighted-f1", fontsize = 20)
plt.savefig("dimension.png", dpi = 300, bbox_inches = 'tight')
plt.clf()
plt.cla()
