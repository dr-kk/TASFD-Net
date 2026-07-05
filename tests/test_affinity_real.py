import numpy as np

from src.clustering.similarity import SimilarityMatrix
from src.clustering.affinity import AffinityCluster

####################################################

feature = np.load(
    "features/accident1.npy"
)

####################################################

similarity = SimilarityMatrix()

S = similarity.cosine(feature)

####################################################

cluster = AffinityCluster()

labels = cluster.fit(S)

print()

print("Number of Clusters")

print(cluster.number_of_clusters())

print()

print("Cluster Centers")

print(cluster.cluster_centers())

print()

print("Cluster Dictionary")

print(cluster.get_cluster_dictionary())