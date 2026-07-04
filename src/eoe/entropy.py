"""
=========================================================
TASFD-Net

Entropy Computation

Author : Dr. Krishan Berwal
=========================================================
"""

import numpy as np


class EntropyCalculator:

    def __init__(self):

        pass

    ##############################################################

    def normalize(self, scores):

        scores = np.asarray(scores, dtype=np.float64)

        scores = scores + 1e-12

        scores = scores / np.sum(scores)

        return scores

    ##############################################################

    def shannon_entropy(self, probabilities):

        probabilities = self.normalize(probabilities)

        entropy = -np.sum(
            probabilities * np.log2(probabilities)
        )

        return entropy

    ##############################################################

    def frame_entropy(self, feature_vector):

        feature_vector = np.abs(feature_vector)

        probability = self.normalize(feature_vector)

        return self.shannon_entropy(probability)

    ##############################################################

    def cluster_entropy(self, features):

        entropy = []

        for feature in features:

            entropy.append(
                self.frame_entropy(feature)
            )

        return np.asarray(entropy)