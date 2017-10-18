# -*- coding:utf-8 -*- 
# Author: Roc-J

import utilities
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# load dataset
filename = 'data_multivar.txt'
X = utilities.load_data(filename)

# set bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# create mean_shift
meanshift_estimator = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_estimator.fit(X)

# labels
labels = meanshift_estimator.labels_

centroids = meanshift_estimator.cluster_centers_
num_clusters = len(np.unique(labels))

print "Number of clusters in input data = ", num_clusters

# visi
plt.figure()

# set markers four
markers = '.*xv'

for i, marker in zip(range(num_clusters), markers):
    # 画出属于某个集群中心的数据点
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='k')

    # plot cluster center
    centroid = centroids[i]
    plt.plot(centroid[0], centroid[1], marker='o', markerfacecolor='k', markeredgecolor='k', markersize=15)

plt.title('Clusters and their centroids')
plt.show()