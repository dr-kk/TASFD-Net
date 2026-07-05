"""
=========================================================
TASFD-Net

Cluster VideoMAE Features

=========================================================
"""

import os
import numpy as np

from src.clustering.affinity import AffinityCluster


class FeatureCluster:

    def __init__(self):

        self.cluster = AffinityCluster()

    #######################################################

    def cluster_feature_file(
            self,
            feature_file):

        features = np.load(feature_file)

        if features.ndim == 1:

            features = features.reshape(1,-1)

        clusters = self.cluster.get_clusters(
            features
        )

        return clusters

    #######################################################

    def cluster_folder(
            self,
            feature_folder):

        results={}

        for file in os.listdir(feature_folder):

            if file.endswith(".npy"):

                path=os.path.join(
                    feature_folder,
                    file
                )

                results[file]=self.cluster_feature_file(
                    path
                )

        return results