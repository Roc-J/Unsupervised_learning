# -*- coding:utf-8 -*- 
# Author: Roc-J

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from utilities import load_data
from itertools import cycle

filename = 'data_perf.txt'
X = load_data(filename)

# 寻找最优参数
eps_grid = np.linspace(0.3, 1.2, num=10)
silhouette_scores = []
eps_best = eps_grid[0]
silhouette_score_max = -1
model_best = None
labels_best = None

# 搜索参数空间
for eps in eps_grid:
    #
    model = DBSCAN(eps=eps, min_samples=5).fit(X)
    labels = model.labels_

    # 性能
    silhouette_score = round(metrics.silhouette_score(X, labels), 4)
    silhouette_scores.append(silhouette_score)

    print "Epsilon: ", eps, "--> silhoustte score:", silhouette_score

    if silhouette_score > silhouette_score_max:
        silhouette_score_max = silhouette_score
        eps_best = eps
        model_best = model
        labels_best = labels

plt.figure()
plt.bar(eps_grid, silhouette_scores, width=0.05, color='k', align='center')
plt.title('Silhouette score vs epslon')

# print the best
print "Best epslon: ", eps_best

model = model_best
labels = labels_best

# 检查标记中没有分配集群的数据点
offset = 0
if -1 in labels:
    offset = 1

num_clusters = len(set(labels)) - offset
print "Estimated number of clusters =", num_clusters

# 从训练模型中提取核心样本的数据索引点
mask_core = np.zeros(labels.shape, dtype=bool)
mask_core[model.core_sample_indices_] = True

# plot the clusters
plt.figure()
labels_uniq = set(labels)
markers = cycle('vo^s<>')

for cur_label, marker in zip(labels_uniq, markers):
    if cur_label == -1:
        marker = '.'

    cur_mask = (labels == cur_label)
    cur_data = X[cur_mask & mask_core]
    plt.scatter(cur_data[:, 0], cur_data[:, 1], marker=marker, s=96, edgecolors='black', facecolor='none')

    cur_data = X[cur_mask & ~mask_core]
    plt.scatter(cur_data[:, 0], cur_data[:, 1], marker=marker, s=32, edgecolors='black')

plt.title('Data separated into clusters')
plt.show()