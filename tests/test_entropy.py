import numpy as np

from src.eoe.entropy import EntropyCalculator

entropy = EntropyCalculator()

###################################################

features = np.random.rand(20,768)

scores = entropy.cluster_entropy(features)

print(scores)

print()

print("Mean :",scores.mean())

print("Max  :",scores.max())

print("Min  :",scores.min())