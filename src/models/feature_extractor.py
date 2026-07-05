"""
=========================================================
TASFD-Net

VideoMAE Feature Extractor

Author : Dr. Krishan Berwal
=========================================================
"""

import os
import torch
import numpy as np

from tqdm import tqdm

from src.models.videomae import VideoMAEExtractor


class FeatureExtractor:

    def __init__(self):

        self.model = VideoMAEExtractor()

    ########################################################

    def extract_dataset(
        self,
        dataset,
        output_folder
    ):

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        self.model.eval()

        with torch.no_grad():

            for sample in tqdm(dataset):

                clips = sample["video"]

                path = sample["path"]

                features = []

                ##################################################

                for clip in clips:

                    clip = clip.unsqueeze(0)

                    feature = self.model.extract_cls(
                        clip
                    )

                    feature = feature.squeeze(0)

                    feature = feature.cpu().numpy()

                    features.append(feature)

                ##################################################

                features = np.asarray(features)

                filename = os.path.basename(path)

                filename = os.path.splitext(filename)[0]

                np.save(
                    os.path.join(
                        output_folder,
                        filename + ".npy"
                    ),
                    features
                )

        print()
        print("Feature Extraction Completed.")