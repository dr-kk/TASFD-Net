import torch

from datasets.dataset import build_dataset

from models.videomae import VideoMAEExtractor

import config

############################################

dataset = build_dataset(
    config.ACCIDENT
)

sample = dataset[0]

video = sample["video"]

video = video.unsqueeze(0)

############################################

model = VideoMAEExtractor()

feature = model.extract_cls(video)

print(feature.shape)