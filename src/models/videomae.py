"""
=========================================================
TASFD-Net

VideoMAE Feature Extractor

Author : Dr Krishan Berwal

=========================================================
"""

import torch
import torch.nn as nn

from transformers import (
    VideoMAEModel,
    VideoMAEImageProcessor
)

import config


class VideoMAEExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        print("Loading VideoMAE...")

        self.processor = VideoMAEImageProcessor.from_pretrained(
            config.MODEL_NAME
        )

        self.model = VideoMAEModel.from_pretrained(
            config.MODEL_NAME
        )

        self.model.eval()

        self.device = torch.device(config.DEVICE)

        self.model.to(self.device)

        print("VideoMAE Loaded Successfully.")

    #########################################################

    @torch.no_grad()
    def forward(self, pixel_values):

        pixel_values = pixel_values.to(self.device)

        outputs = self.model(
            pixel_values=pixel_values
        )

        return outputs.last_hidden_state

    #########################################################

    @torch.no_grad()
    def extract_cls(self, pixel_values):

        features = self.forward(pixel_values)

        cls = features[:, 0]

        return cls

    #########################################################

    @torch.no_grad()
    def extract_mean(self, pixel_values):

        features = self.forward(pixel_values)

        mean = features.mean(dim=1)

        return mean