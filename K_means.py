from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

x,y = make_blobs(n_samples = 560, centers = 4 , random_state = 42)

km = KMeans(n_clusters = 4, random_state = 42)

km.fit(x)

centers = km.cluster_centers_
labels = km.labels_

plt.scatter(x[:,0], x[:,1], c = labels, cmap="viridis", alpha = 0.6)
plt.scatter(centers[:,0], centers[:,1], c="red" , s=200, marker="x", label = "Merkezler")
plt.legend()
plt.show()
