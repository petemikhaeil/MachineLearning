import numpy as np
from sklearn.cluster import MeanShift as ms
from sklearn.cluster import AffinityPropagation as ap
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
centers = [[1, 1, 1], [5, 5, 5], [10, 11, 5]]

X, _ = make_blobs(n_samples=50, centers=centers, cluster_std=1, n_features=3)


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)


labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

n_clusters = len(np.unique(labels))


colors = ['r', 'g', 'b']
for i in range(len(X)):
	ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]])

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='black')

plt.show()

