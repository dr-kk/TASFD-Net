from src.datasets.dataset import build_dataset

import config

dataset = build_dataset(

    config.ACCIDENT

)

print(len(dataset))

sample = dataset[0]

print(sample["video"].shape)

print(sample["path"])