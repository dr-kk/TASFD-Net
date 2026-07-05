"""
=========================================================
TASFD-Net

Main Pipeline

Author : Dr. Krishan Berwal
=========================================================
"""

import os
import numpy as np

from src.datasets.dataset import build_dataset
from src.models.feature_extractor import FeatureExtractor
from src.clustering.similarity import SimilarityMatrix
from src.clustering.affinity import AffinityCluster
from src.eoe.cluster_entropy import ClusterEntropy
from src.eoe.eoe_score import EOEScore
from src.selection.superframe_selector import SuperframeSelector
from src.evaluation.summary_generator import SummaryGenerator


class TASFDNet:

    def __init__(self):

        self.extractor = FeatureExtractor()

        self.similarity = SimilarityMatrix()

        self.cluster = AffinityCluster()

        self.entropy = ClusterEntropy()

        self.eoe = EOEScore()

        self.selector = SuperframeSelector()

        self.generator = SummaryGenerator()

    ########################################################

    def summarize(
            self,
            video_path,
            feature_folder,
            output_file):

        dataset = build_dataset(video_path)

        self.extractor.extract_dataset(
            dataset,
            feature_folder
        )

        feature_file = os.path.join(
            feature_folder,
            os.path.splitext(
                os.path.basename(video_path)
            )[0] + ".npy"
        )

        features = np.load(feature_file)

        S = self.similarity.cosine(features)

        self.cluster.fit(S)

        clusters = self.cluster.get_cluster_dictionary()

        entropy_scores = self.entropy.compute(
            features,
            clusters
        )

        eoe_scores = self.eoe.compute(
            entropy_scores,
            clusters
        )

        summary = self.selector.select(
            features,
            clusters,
            eoe_scores,
            summary_ratio=0.15
        )

        clips = [x["clip"] for x in summary]

        self.generator.generate(
            video_path,
            clips,
            output_file
        )

        return summary