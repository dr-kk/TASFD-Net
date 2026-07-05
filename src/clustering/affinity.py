"""
=========================================================
TASFD-Net

Affinity Propagation Clustering

Author : Dr. Krishan Berwal
=========================================================
"""

import numpy as np
from sklearn.cluster import AffinityPropagation


class AffinityCluster:

    def __init__(
        self,
        damping=0.90,
        preference=None,
        random_state=42
    ):

        self.damping = damping
        self.preference = preference
        self.random_state = random_state

    #########################################################

    def fit(self, similarity_matrix):

        if self.preference is None:

            # Standard AP initialization
            preference = np.median(similarity_matrix)

        else:

            preference = self.preference

        model = AffinityPropagation(

            affinity="precomputed",

            damping=self.damping,

            preference=preference,

            random_state=self.random_state

        )

        model.fit(similarity_matrix)

        self.labels = model.labels_

        self.centers = model.cluster_centers_indices_

        return self.labels

    #########################################################

    def cluster_centers(self):

        return self.centers

    #########################################################

    def number_of_clusters(self):

        return len(self.centers)

    #########################################################

    def get_cluster_dictionary(self):

        clusters = {}

        for idx, label in enumerate(self.labels):

            if label not in clusters:

                clusters[label] = []

            clusters[label].append(idx)

        return clusters