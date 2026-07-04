import numpy as np

from src.clustering.affinity import AffinityCluster

#########################################

features = np.random.rand(100,768)

cluster = AffinityCluster()

labels = cluster.fit(features)

print(labels)

print("Clusters :",cluster.n_clusters())

print(cluster.cluster_centers())