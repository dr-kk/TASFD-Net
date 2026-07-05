"""
=========================================================
TASFD-Net

Run Complete Accident Dataset

Author : Dr. Krishan Berwal
=========================================================
"""

import os
import glob
import numpy as np

from src.dataset import build_dataset

from src.features.extractor import FeatureExtractor

from src.clustering.similarity import SimilarityMatrix
from src.clustering.affinity import AffinityCluster

from src.eoe.cluster_entropy import ClusterEntropy
from src.eoe.eoe_score import EOEScore

from src.selection.superframe_selector import SuperframeSelector
from src.evaluation.summary_generator import SummaryGenerator


DATASET = "./datasets/Accident"

FEATURE_DIR = "./features"

SUMMARY_DIR = "./results"


os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


#########################################################

print("=" * 60)
print("TASFD-Net")
print("Processing Accident Dataset")
print("=" * 60)

#########################################################

videos = sorted(

    glob.glob(

        os.path.join(DATASET, "*.mp4")

    )

)

#########################################################

extractor = FeatureExtractor()

#########################################################

for video in videos:

    print()

    print("------------------------------------------")

    print("Video :", os.path.basename(video))

    #####################################################

    dataset = build_dataset(video)

    extractor.extract_dataset(

        dataset,

        FEATURE_DIR

    )

    #####################################################

    feature_file = os.path.join(

        FEATURE_DIR,

        os.path.splitext(

            os.path.basename(video)

        )[0] + ".npy"

    )

    features = np.load(feature_file)

    #####################################################

    similarity = SimilarityMatrix()

    S = similarity.cosine(features)

    #####################################################

    cluster = AffinityCluster()

    cluster.fit(S)

    clusters = cluster.get_cluster_dictionary()

    #####################################################

    entropy = ClusterEntropy()

    entropy_scores = entropy.compute(

        features,

        clusters

    )

    #####################################################

    eoe = EOEScore()

    scores = eoe.compute(

        entropy_scores,

        clusters

    )

    #####################################################

    selector = SuperframeSelector()

    summary = selector.select(

        features,

        clusters,

        scores,

        summary_ratio=0.15

    )

    #####################################################

    clips = [

        x["clip"]

        for x in summary

    ]

    #####################################################

    output = os.path.join(

        SUMMARY_DIR,

        os.path.basename(video)

    )

    SummaryGenerator().generate(

        video,

        clips,

        output

    )

    print("Completed :", os.path.basename(video))

#########################################################

print()

print("=" * 60)
print("ALL VIDEOS PROCESSED SUCCESSFULLY")
print("=" * 60)