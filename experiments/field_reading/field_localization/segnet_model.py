"""Compact fully-convolutional field-mask net: MobileNetV3-Large encoder + FPN-style decoder.

Input: (N, 3, CANVAS_HEIGHT_PX, CANVAS_WIDTH_PX) float RGB in [0, 1] (normalization is inside the
model so the browser only divides by 255). Output: (N, 7, H/2, W/2) logits, one filled-box mask per
target field, channel order = segnet_config.FIELD_CHANNEL_NAMES.

- Only Conv / BN / HardSwish / ReLU / bilinear x2 Resize / Add: every op exports to ONNX opset 17
  and runs in onnxruntime-web (wasm / webgpu).
"""

import timm
import torch
from torch import nn
from torch.nn import functional

from experiments.field_reading.field_localization.segnet_config import (DECODER_CHANNELS, ENCODER_NAME,
                                                                        FIELD_CHANNEL_NAMES, IMAGENET_MEAN_RGB,
                                                                        IMAGENET_STD_RGB)


def conv_bn_relu(input_channels: int, output_channels: int) -> nn.Sequential:
    """3x3 conv + batch norm + ReLU."""
    return nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))


class FieldMaskSegmentationNet(nn.Module):
    """Predicts one filled-box logit map per check field at stride 2 of the input canvas."""

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = timm.create_model(ENCODER_NAME, pretrained=pretrained_encoder, features_only=True)
        encoder_channels = self.encoder.feature_info.channels()   # strides 2, 4, 8, 16, 32
        self.lateral_projections = nn.ModuleList(
            [nn.Conv2d(channels, DECODER_CHANNELS, 1) for channels in encoder_channels])
        self.merge_blocks = nn.ModuleList(
            [conv_bn_relu(DECODER_CHANNELS, DECODER_CHANNELS) for _ in encoder_channels[:-1]])
        self.mask_head = nn.Sequential(conv_bn_relu(DECODER_CHANNELS, DECODER_CHANNELS // 2),
                                       nn.Conv2d(DECODER_CHANNELS // 2, len(FIELD_CHANNEL_NAMES), 1))
        self.register_buffer("input_mean", torch.tensor(IMAGENET_MEAN_RGB).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(IMAGENET_STD_RGB).view(1, 3, 1, 1))

    def forward(self, canvas_rgb: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) in [0, 1] -> (N, 7, H/2, W/2) logits."""
        encoder_features = self.encoder((canvas_rgb - self.input_mean) / self.input_std)
        laterals = [projection(feature) for projection, feature in zip(self.lateral_projections, encoder_features)]
        decoded = laterals[-1]
        for level_index in range(len(laterals) - 2, -1, -1):
            decoded = functional.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
            decoded = self.merge_blocks[level_index](decoded + laterals[level_index])
        return self.mask_head(decoded)
