# -*- coding:utf-8 -*- 
# Author: Roc-J

import utilities
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
# load data
filename = 'data_perf.txt'
data = utilities.load_data(filename)

# 迭代集群数来找最好的
scores = []
range_values = np.arange(2, 10)

for i in range_values:
    # 训练模型
    kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
    kmeans.fit(data)
    score = metrics.silhouette_score(data, kmeans.labels_, metric='euclidean', sample_size=len(data))

    print "\nNumber of clusters = ", i
    print "Silhouette score = ", score

    scores.append(score)

# plot clusters vs scores
plt.figure()
plt.bar(range_values, scores, width=0.6, color='k', align='center')
plt.title('Silhouette score vs number of clusters')

#plot the data
plt.figure()
plt.scatter(data[:, 0], data[:, 1], color='k', s=30, marker='o', edgecolors='k', facecolor='none')
plt.title('Input data')
plt.show()