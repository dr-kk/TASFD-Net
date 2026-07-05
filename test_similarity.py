import numpy as np

from src.clustering.similarity import SimilarityMatrix

###########################################

feature=np.load(

    "features/accident1.npy"

)

###########################################

sim=SimilarityMatrix()

S=sim.cosine(feature)

print(S.shape)

print()

print(S)