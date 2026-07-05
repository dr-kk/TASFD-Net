"""
=========================================================
TASFD-Net

Dataset Loader

Author : Dr Krishan Berwal
=========================================================
"""

import os
import torch

from torch.utils.data import Dataset

from transformers import VideoMAEImageProcessor

from src.utils.video_utils import (
    read_video,
    resize_frames,
    create_clips
)

import config


class BaseVideoDataset(Dataset):

    def __init__(
        self,
        root_dir,
        processor=None,
        clip_length=16,
        stride=4
    ):

        self.root_dir = root_dir
        self.processor = processor
        self.clip_length = clip_length
        self.stride = stride

        self.video_files = []

        self.load_videos()

    ####################################################

    def load_videos(self):

        exts = [".mp4", ".avi", ".mov", ".mpg"]

    # Case 1: Single video file
        if os.path.isfile(self.root_dir):

            if os.path.splitext(self.root_dir)[1].lower() in exts:

                self.video_files.append(self.root_dir)

            return

    # Case 2: Folder of videos
        for root, _, files in os.walk(self.root_dir):

            for file in files:

                if os.path.splitext(file)[1].lower() in exts:

                    self.video_files.append(
                        os.path.join(root, file)
                    )

    ####################################################

    def __len__(self):

        return len(self.video_files)

    ####################################################

    def __getitem__(self, index):

        video_path = self.video_files[index]

        ################################################

        frames = read_video(video_path)

        frames = resize_frames(
            frames,
            config.IMG_SIZE
        )

        ################################################

        clips = create_clips(
            frames,
            clip_length=self.clip_length,
            stride=self.stride
        )

        ################################################

        if len(clips) == 0:

            clips = [frames]

        ################################################
        total_clips = len(clips)

        MAX_CLIPS = min(
            total_clips,
            max(
                300,
                int(total_clips * 0.25)
            )
        )

        processed = []

        for clip in clips[:MAX_CLIPS]:

            if self.processor is not None:

                clip = self.processor(
                    clip,
                    return_tensors="pt"
                )["pixel_values"][0]

            processed.append(clip)

        ################################################

        ################################################

        return {

            "video": processed,

            "path": video_path

        }


########################################################

def build_dataset(dataset_path):

    processor = VideoMAEImageProcessor.from_pretrained(
        config.MODEL_NAME
    )

    dataset = BaseVideoDataset(
        dataset_path,
        processor=processor
    )

    return dataset