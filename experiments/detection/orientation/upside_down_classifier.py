"""Tiny CNN that says whether a landscape check crop is upside down.

Input: (B, 1, ORIENTATION_CROP_HEIGHT, ORIENTATION_CROP_WIDTH) float in [0, 1]
(grayscale crop from `rectify_check_crop.warp_quadrilateral_to_crop`).
Output: (B,) logit, > 0 means upside down.

Plain PyTorch, trained from scratch, so the weights carry no third-party licence and
can ship in the public app (ONNX, a few hundred KB). Global average pooling makes it
tolerant of small crop misalignment; the cue it learns is page layout (MICR band at the
bottom, bank/payer block at the top, text direction), which survives a 224x96 crop.
"""

import torch
from torch import nn

CLASSIFIER_CHANNEL_WIDTHS = (16, 32, 64, 96, 128)


def convolution_block(input_channels: int, output_channels: int) -> nn.Sequential:
    """3x3 stride-2 conv, batch norm, ReLU, then a 3x3 stride-1 conv, batch norm, ReLU."""
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )


class UpsideDownCheckClassifier(nn.Module):
    """Binary upright (logit < 0) vs upside-down (logit > 0) classifier for check crops."""

    def __init__(self):
        super().__init__()
        blocks = []
        input_channels = 1
        for output_channels in CLASSIFIER_CHANNEL_WIDTHS:
            blocks.append(convolution_block(input_channels, output_channels))
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.classifier_head = nn.Linear(input_channels, 1)

    def forward(self, crop_batch: torch.Tensor) -> torch.Tensor:
        """Return one logit per crop."""
        pooled_features = self.features(crop_batch).mean(dim=(2, 3))
        return self.classifier_head(pooled_features).squeeze(1)
