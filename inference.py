"""
TASFD-Net Inference
"""

import argparse
import os

from src.models.tasfdnet import TASFDNet

parser = argparse.ArgumentParser()

parser.add_argument(
    "--video",
    required=True
)

args = parser.parse_args()

os.makedirs("./results/inference", exist_ok=True)
os.makedirs("./results/features", exist_ok=True)

model = TASFDNet()

outfile = os.path.join(
    "./results/inference",
    os.path.basename(args.video)
)

model.summarize(

    video_path=args.video,

    feature_folder="./results/features",

    output_file=outfile

)

print()

print("Inference Completed")

print(outfile)