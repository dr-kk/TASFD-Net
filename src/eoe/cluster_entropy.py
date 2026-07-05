"""
=========================================================
TASFD-Net

Cluster-wise Entropy

=========================================================
"""

import numpy as np


class ClusterEntropy:

    def __init__(self):
        pass

    #########################################################

    def compute(self, features, cluster_dict):

        entropy_scores = {}

        for cluster_id, indices in cluster_dict.items():

            cluster_features = features[indices]

            centroid = np.mean(cluster_features, axis=0)

            distances = np.linalg.norm(
                cluster_features - centroid,
                axis=1
            )

            probabilities = distances + 1e-12
            probabilities = probabilities / probabilities.sum()

            entropy = float(
                -np.sum(
                    probabilities *
                    np.log2(probabilities)
                )
            )

            if abs(entropy) < 1e-10:
                entropy = 0.0

            entropy_scores[cluster_id] = entropy

        return entropy_scores