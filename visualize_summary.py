"""
=========================================================
TASFD-Net

Summary Visualization

Author : Dr. Krishan Berwal
=========================================================
"""

import cv2
import os

VIDEO = "./datasets/Accident/accident1.mp4"
SUMMARY = "./results/accident1.mp4"

os.makedirs("./results/figures", exist_ok=True)

#########################################################

def save_frame(video, frame_no, output):

    cap = cv2.VideoCapture(video)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    ret, frame = cap.read()

    if ret:

        cv2.imwrite(output, frame)

    cap.release()

#########################################################

cap = cv2.VideoCapture(VIDEO)

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

save_frame(
    VIDEO,
    int(frames*0.25),
    "./results/figures/original_25.jpg"
)

save_frame(
    VIDEO,
    int(frames*0.50),
    "./results/figures/original_50.jpg"
)

save_frame(
    VIDEO,
    int(frames*0.75),
    "./results/figures/original_75.jpg"
)

cap = cv2.VideoCapture(SUMMARY)

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

save_frame(
    SUMMARY,
    int(frames*0.25),
    "./results/figures/summary_25.jpg"
)

save_frame(
    SUMMARY,
    int(frames*0.50),
    "./results/figures/summary_50.jpg"
)

save_frame(
    SUMMARY,
    int(frames*0.75),
    "./results/figures/summary_75.jpg"
)

print()

print("Figures Saved")

print("./results/figures")