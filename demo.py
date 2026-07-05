"""
=========================================================
TASFD-Net

Demo

Author : Dr. Krishan Berwal
=========================================================
"""

import os
import argparse

from src.models.tasfdnet import TASFDNet


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video"
    )

    args = parser.parse_args()

    feature_folder = "./results/features"
    output_folder = "./results/demo"

    os.makedirs(feature_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    output_video = os.path.join(
        output_folder,
        os.path.basename(args.video)
    )

    model = TASFDNet()

    summary = model.summarize(
        video_path=args.video,
        feature_folder=feature_folder,
        output_file=output_video
    )

    print()
    print("=" * 60)
    print("TASFD-Net Demo Completed")
    print("=" * 60)
    print("Input Video :", args.video)
    print("Summary     :", output_video)
    print("Superframes :", len(summary))
    print("=" * 60)


if __name__ == "__main__":
    main()