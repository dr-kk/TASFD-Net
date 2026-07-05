"""
=========================================================
TASFD-Net

Evaluation Metrics

Author : Dr. Krishan Berwal
=========================================================
"""

import numpy as np


class Evaluation:

    def precision(self, pred, gt):

        pred = np.asarray(pred).astype(bool)
        gt = np.asarray(gt).astype(bool)

        tp = np.sum(pred & gt)
        fp = np.sum(pred & (~gt))

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    #######################################################

    def recall(self, pred, gt):

        pred = np.asarray(pred).astype(bool)
        gt = np.asarray(gt).astype(bool)

        tp = np.sum(pred & gt)
        fn = np.sum((~pred) & gt)

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    #######################################################

    def fscore(self, p, r):

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)