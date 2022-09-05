import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data = pd.read_csv("C:/Users/admin/Downloads/Iris.csv", skiprows=[0],
                   names=['sepal length', 'sepal width', 'petal length', 'petal width', 'species'])
data.drop(data.columns[[4]], axis=1, inplace=True)
# print(data)

##first applying PCA on our data

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = data.loc[:, features].values

x = StandardScaler().fit_transform(x)
data_df = pd.DataFrame(x, columns=features)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(principal_components, columns=['principal_component_1', 'principal_component_2'])
print(principal_df)
#
# find optimal value of K to k means clustering
K = np.arange(1, 10)
sse = []
for k in K:
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans = kmeans.fit(principal_df)
    sse_k = kmeans.inertia_  # .inertia_ calculated by measuring the distance between each data point and its centroid.
    sse.append(sse_k)

plt.xlabel("Number of Cluster")
plt.ylabel("SSE")
plt.plot(K,sse)
plt.savefig(r"D:\code\k_means_elbow_method\test0")
# plt.show()
# so here elbow point is at 3rd position so optimal clusters are 3
# k=3

kmeans = cluster.KMeans(n_clusters=3)
kmeans = kmeans.fit(principal_components)
principal_df['Clusters'] = kmeans.labels_
clusters = principal_df["Clusters"]
principal_df = principal_df.to_numpy()
print(principal_df)

list_of_colours = []
for cluster in clusters:
    if cluster == 0:
        list_of_colours.append("red")
    elif cluster == 1:
        list_of_colours.append("blue")
    else:
        list_of_colours.append('yellow')
    print(list_of_colours)

plt.scatter(principal_df[:, 0], principal_df[:, 1], c=list_of_colours)
plt.savefig(r"D:\code\k_means_clustering\test")
# plt.show()
