import numpy as np

import config

from src.clustering.similarity import SimilarityMatrix
from src.clustering.affinity import AffinityCluster

from src.eoe.cluster_entropy import ClusterEntropy
from src.eoe.eoe_score import EOEScore

from src.selection.superframe_selector import SuperframeSelector

from src.evaluation.summary_generator import SummaryGenerator

############################################################

features=np.load(

    "features/accident1.npy"

)

############################################################

similarity=SimilarityMatrix()

S=similarity.cosine(features)

############################################################

cluster=AffinityCluster()

cluster.fit(S)

clusters=cluster.get_cluster_dictionary()

############################################################

entropy=ClusterEntropy()

entropy_scores=entropy.compute(

    features,

    clusters

)

############################################################

eoe=EOEScore()

scores=eoe.compute(

    entropy_scores,

    clusters

)

############################################################

selector=SuperframeSelector()

summary=selector.select(

    features,

    clusters,

    scores,

    summary_ratio=0.15

)

############################################################

clips=[]

for item in summary:

    clips.append(

        item["clip"]

    )

############################################################

generator=SummaryGenerator()

generator.generate(

    "./datasets/Accident/accident1.mp4",

    clips,

    "./results/summary.mp4"

)