import numpy as np

from src.eoe.eoe import EOECalculator

##################################################

probability = np.random.rand(20)

entropy = np.random.rand(20)*10

##################################################

model = EOECalculator()

eoe = model.compute(
    probability,
    entropy
)

model.statistics(eoe)

print()

print("Threshold")

print(model.median_threshold(eoe))

print()

print("Selected Frames")

print(model.select_frames(eoe))