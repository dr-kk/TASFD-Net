import numpy as np

from src.clustering.similarity import SimilarityMatrix
from src.clustering.affinity import AffinityCluster
from src.eoe.cluster_entropy import ClusterEntropy

########################################################

features = np.load("features/accident1.npy")

########################################################

similarity = SimilarityMatrix()

S = similarity.cosine(features)

########################################################

cluster = AffinityCluster()

cluster.fit(S)

clusters = cluster.get_cluster_dictionary()

########################################################

entropy = ClusterEntropy()

scores = entropy.compute(
    features,
    clusters
)

print()

for k, v in scores.items():

    print(k, ":", round(v, 4))