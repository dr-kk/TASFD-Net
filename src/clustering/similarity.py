"""
=========================================================
TASFD-Net

Similarity Matrix

=========================================================
"""

import numpy as np


class SimilarityMatrix:

    #######################################################

    def cosine(self,features):

        norm=np.linalg.norm(
            features,
            axis=1,
            keepdims=True
        )

        features=features/norm

        similarity=np.matmul(
            features,
            features.T
        )

        return similarity

    #######################################################

    def euclidean(self,features):

        distance=np.sum(

            (features[:,None]-features)**2,

            axis=2

        )

        similarity=-distance

        return similarity