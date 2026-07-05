import config

from src.datasets.dataset import build_dataset

from src.models.feature_extractor import FeatureExtractor

##########################################################

dataset = build_dataset(

    config.ACCIDENT

)

##########################################################

extractor = FeatureExtractor()

extractor.extract_dataset(

    dataset,

    "features"

)