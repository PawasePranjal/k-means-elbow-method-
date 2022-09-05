import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

data = pd.read_csv("C:/Users/admin/Downloads/Iris.csv", skiprows=[0],
                   names=['sepal length', 'sepal width', 'petal length', 'petal width', 'species'])
data.drop(data.columns[[4]], axis=1, inplace=True)
# print(data)
# print(type(data))

sse = []
# find optimal value of K to k means clustering
K = np.arange(1, 10)
sse = []
for k in K:
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans = kmeans.fit(data)
    sse_k = kmeans.inertia_
    sse.append(sse_k)

plt.xlabel("Number of Cluster")
plt.ylabel("SSE")
plt.plot(K, sse)
plt.savefig(r"D:\code\k_means_elbow_method\test0")
