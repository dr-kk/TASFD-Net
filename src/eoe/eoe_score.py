"""
=========================================================
TASFD-Net

Entropy of Events (EOE)

Author : Dr. Krishan Berwal
=========================================================
"""

import numpy as np


class EOEScore:

    def __init__(self, alpha=0.6):

        self.alpha = alpha

    ######################################################

    def normalize(self, values):

        values = np.asarray(values, dtype=np.float64)

        minimum = values.min()
        maximum = values.max()

        if maximum - minimum < 1e-12:
            return np.zeros_like(values)

        return (values - minimum) / (maximum - minimum)

    ######################################################

    def compute(self,
                entropy_scores,
                cluster_dict):

        entropy = np.array(
            list(entropy_scores.values())
        )

        entropy = self.normalize(entropy)

        cluster_sizes = np.array(

            [len(cluster_dict[k])
             for k in entropy_scores.keys()],

            dtype=np.float64

        )

        cluster_sizes = self.normalize(cluster_sizes)

        eoe = (

            self.alpha * entropy +

            (1 - self.alpha) * cluster_sizes

        )

        return dict(

            zip(

                entropy_scores.keys(),

                eoe

            )

        )