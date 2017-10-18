# -*- coding:utf-8 -*- 
# Author: Roc-J

import numpy as np
import utilities
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# load dataset
filename = 'data_multivar.txt'
data = utilities.load_data(filename)

# plot the data
plt.figure()
plt.scatter(data[:, 0], data[:, 1], facecolor='None', edgecolors='k', marker='o', s=30)
x_min, x_max = min(data[:, 0]) - 1.0, max(data[:, 0]) + 1.0
y_min, y_max = min(data[:, 1]) - 1.0, max(data[:, 1]) + 1.0
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('Input data')
plt.show()

# create kmean model
num_clusters = 4
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
kmeans.fit(data)

# set size of the mesh
step_size = 0.01
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

# predict labels for all points in the mesh
predicted_labels = kmeans.predict(np.c_[x_values.ravel(), y_values.ravel()])
predicted_labels = predicted_labels.reshape(x_values.shape)

# plot the result
plt.figure()
plt.clf()
plt.imshow(predicted_labels, interpolation='nearest', extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')
plt.scatter(data[:, 0], data[:, 1], facecolor='None', edgecolors='k', marker='o', s=30)
# plot the cluster centers points
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, linewidths=3, color='k', zorder=10, facecolor='black')
plt.title('Centoids and boundaries obtained useing kmeans.')
plt.show()