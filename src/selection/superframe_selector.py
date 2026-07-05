"""
=========================================================
TASFD-Net

Superframe Selector

Author : Dr. Krishan Berwal
=========================================================
"""

import numpy as np
from torch.return_types import topk


class SuperframeSelector:

    def __init__(self):
        pass

    #########################################################

    def representative_clip(
            self,
            features,
            indices):

        cluster = features[indices]

        centroid = np.mean(cluster, axis=0)

        distance = np.linalg.norm(

            cluster-centroid,

            axis=1

        )

        idx = np.argmin(distance)

        return indices[idx]

    #########################################################

    def select(
            self,
            features,
            cluster_dict,
            eoe_scores,
            summary_ratio=0.15):

        representatives = []

        #####################################################

        for cluster_id in sorted(

                eoe_scores,

                key=eoe_scores.get,

                reverse=True):

            clip = self.representative_clip(

                features,

                cluster_dict[cluster_id]

            )

            representatives.append(

                {

                    "cluster":cluster_id,

                    "clip":clip,

                    "score":eoe_scores[cluster_id]

                }

            )

        #####################################################

        representatives.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        total = len(representatives)

# Always keep at least 5 superframes
        topk = max(
            5,
            min(20, int(total * summary_ratio))
        )

# Don't exceed available representatives
        topk = min(topk, total)

        representatives = representatives[:topk]

        return representatives