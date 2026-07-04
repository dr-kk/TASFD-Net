"""
=========================================================
TASFD-Net

Affinity Propagation Clustering

Author : Dr Krishan Berwal
=========================================================
"""

import numpy as np
from sklearn.cluster import AffinityPropagation


class AffinityCluster:

    def __init__(self,
                 damping=0.90,
                 preference=None,
                 random_state=42):

        self.model = AffinityPropagation(
            damping=damping,
            preference=preference,
            random_state=random_state
        )

    #########################################################

    def fit(self, features):

        """
        Parameters
        ----------
        features : ndarray
            Shape = (N,768)
        """

        self.model.fit(features)

        return self.model.labels_

    #########################################################

    def cluster_centers(self):

        return self.model.cluster_centers_indices_

    #########################################################

    def n_clusters(self):

        return len(self.model.cluster_centers_indices_)

    #########################################################

    def predict(self, features):

        return self.model.predict(features)

    #########################################################

    def get_clusters(self, features):

        labels = self.fit(features)

        clusters = {}

        for i, label in enumerate(labels):

            if label not in clusters:
                clusters[label] = []

            clusters[label].append(i)

        return clusters