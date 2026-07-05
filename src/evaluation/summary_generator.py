"""
=========================================================
TASFD-Net

Summary Generator

Author : Dr. Krishan Berwal
=========================================================
"""

import cv2
import os


class SummaryGenerator:

    def __init__(self):
        pass

    #########################################################

    def generate(self,
                 video_path,
                 clips,
                 output_file):

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(

            output_file,

            cv2.VideoWriter_fourcc(*'mp4v'),

            fps,

            (width,height)

        )

        frame_id = 0

        clip_length = 16

        selected = set()

        for clip in clips:

            start = clip * 4

            end = start + clip_length

            for i in range(start,end):

                selected.add(i)

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            if frame_id in selected:

                writer.write(frame)

            frame_id += 1

        cap.release()

        writer.release()

        print()

        print("Summary Saved")

        print(output_file)