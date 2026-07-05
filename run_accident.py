"""
=========================================================
TASFD-Net

Run Complete Accident Dataset

Author : Dr. Krishan Berwal
=========================================================
"""

import os
import glob
import time

import config

from src.models.tasfdnet import TASFDNet


#########################################################
# Output folders
#########################################################

FEATURE_FOLDER = "./results/features"
OUTPUT_FOLDER = "./results/summaries"

os.makedirs(FEATURE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#########################################################
# Load TASFD-Net
#########################################################

print("=" * 60)
print("Loading TASFD-Net...")
print("=" * 60)

model = TASFDNet()

#########################################################
# Collect videos
#########################################################

videos = sorted(

    glob.glob(

        os.path.join(
            config.ACCIDENT,
            "*.mp4"
        )

    )

)

print()
print("=" * 60)
print("Processing Accident Dataset")
print("Total Videos :", len(videos))
print("=" * 60)

#########################################################

processed = 0
skipped = 0

#########################################################

for idx, video in enumerate(videos, start=1):

    print()
    print("=" * 60)
    print(f"[{idx}/{len(videos)}]")
    print("Video :", os.path.basename(video))
    print("=" * 60)

    output_file = os.path.join(

        OUTPUT_FOLDER,

        os.path.basename(video)

    )

    start_time = time.time()

    try:

        model.summarize(

            video_path=video,

            feature_folder=FEATURE_FOLDER,

            output_file=output_file

        )

        elapsed = time.time() - start_time

        print()

        print("Status  : SUCCESS")
        print("Runtime :", round(elapsed,2), "seconds")

        processed += 1

    except Exception as e:

        elapsed = time.time() - start_time

        print()

        print("Status  : FAILED")
        print("Runtime :", round(elapsed,2), "seconds")
        print("Reason  :", e)

        skipped += 1

#########################################################

print()
print("=" * 60)
print("TASFD-Net Execution Completed")
print("=" * 60)

print("Videos Found     :", len(videos))
print("Processed        :", processed)
print("Skipped          :", skipped)
print("Summary Folder   :", OUTPUT_FOLDER)
print("Feature Folder   :", FEATURE_FOLDER)

print("=" * 60)

with open("./results/logs/run_log.txt", "a") as f:

    f.write(
        f"{os.path.basename(video)}, "
        f"{elapsed:.2f} sec\n"
    )