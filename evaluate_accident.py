"""
=========================================================
TASFD-Net

Accident Dataset Evaluation

Author : Dr. Krishan Berwal
=========================================================
"""

import os
import glob
import cv2
import pandas as pd

import config


RESULT_FOLDER = "./results"
OUTPUT_CSV = "./results/statistics.csv"


#########################################################

def count_frames(video):

    cap = cv2.VideoCapture(video)

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    duration = 0

    if fps > 0:

        duration = frames / fps

    cap.release()

    return frames, fps, duration


#########################################################

rows = []

videos = sorted(

    glob.glob(

        os.path.join(

            config.ACCIDENT,

            "*.mp4"

        )

    )

)

#########################################################

print("=" * 60)
print("Evaluating Accident Dataset")
print("=" * 60)

for video in videos:

    name = os.path.basename(video)

    summary = os.path.join(

        RESULT_FOLDER,

        name

    )

    if not os.path.exists(summary):

        print("Summary Missing :", name)

        continue

    #####################################################

    total_frames, fps, duration = count_frames(video)

    summary_frames, _, _ = count_frames(summary)

    #####################################################

    compression = 0

    if total_frames > 0:

        compression = (

            summary_frames /

            total_frames

        ) * 100

    #####################################################

    rows.append({

        "Video": name,

        "Original Frames": total_frames,

        "Summary Frames": summary_frames,

        "Compression (%)": round(compression,2),

        "FPS": round(fps,2),

        "Duration (sec)": round(duration,2)

    })

#########################################################

df = pd.DataFrame(rows)

df.to_csv(

    OUTPUT_CSV,

    index=False

)

#########################################################

print()

print(df)

print()

print("=" * 60)

print("Statistics Saved")

print(OUTPUT_CSV)

print("=" * 60)