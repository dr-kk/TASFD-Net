"""
=========================================================
TASFD-Net

Entropy of Events (EOE)

Author : Dr Krishan Berwal
=========================================================
"""

import numpy as np


class EOECalculator:

    def __init__(self,
                 alpha=0.60):

        self.alpha = alpha

    #########################################################

    def compute(self,
                probability,
                entropy):

        probability = np.asarray(probability)

        entropy = np.asarray(entropy)

        eoe = self.alpha * probability + \
              (1-self.alpha) * entropy

        return eoe

    #########################################################

    def median_threshold(
            self,
            eoe):

        return np.median(eoe)

    #########################################################

    def select_frames(
            self,
            eoe):

        threshold = self.median_threshold(eoe)

        selected = np.where(
            eoe >= threshold
        )[0]

        return selected

    #########################################################

    def statistics(
            self,
            eoe):

        print()

        print("EOE Mean :",eoe.mean())

        print("EOE Max  :",eoe.max())

        print("EOE Min  :",eoe.min())