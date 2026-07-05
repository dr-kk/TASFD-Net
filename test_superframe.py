import numpy as np

from src.clustering.similarity import SimilarityMatrix
from src.clustering.affinity import AffinityCluster
from src.eoe.cluster_entropy import ClusterEntropy
from src.eoe.eoe_score import EOEScore
from src.selection.superframe_selector import SuperframeSelector

##########################################################

features = np.load(
    "features/accident1.npy"
)

##########################################################

similarity = SimilarityMatrix()

S = similarity.cosine(features)

##########################################################

cluster = AffinityCluster()

cluster.fit(S)

clusters = cluster.get_cluster_dictionary()

##########################################################

entropy = ClusterEntropy()

entropy_scores = entropy.compute(
    features,
    clusters
)

##########################################################

eoe = EOEScore()

scores = eoe.compute(
    entropy_scores,
    clusters
)

##########################################################

selector = SuperframeSelector()

summary = selector.select(

    features,

    clusters,

    scores, 
    summary_ratio=0.15

)

##########################################################

print()

print("Representative Superframes")

print("-------------------------------------------")

for item in summary:

    print(

        f"Cluster {item['cluster']:2d}"

        f"   Clip {item['clip']:3d}"

        f"   Score {item['score']:.4f}"

    )