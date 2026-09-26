"""CRNN line recognizer (conv feature extractor -> BiLSTM -> per-timestep CTC classes) and its variants.

The conv stack reduces height to 1 and width by 4, so timesteps = ceil-ish(width / 4)
(`timesteps_for_width`). Output is log-softmax (B, T, C). Everything is ONNX-exportable with a
dynamic batch and width axis. Variants are registered by name so configs/CLIs pick them by id.
"""

from dataclasses import dataclass

import torch
from torch import nn

WIDTH_DOWNSAMPLE_FACTOR = 4


@dataclass(frozen=True)
class CrnnVariant:
    """Hyper-parameters of one CRNN size."""

    input_height: int
    conv_channels: tuple[int, int, int, int]
    lstm_hidden_size: int
    lstm_layers: int
    min_width: int
    max_width: int


CRNN_VARIANT_REGISTRY: dict[str, CrnnVariant] = {
    "crnn_small_h32": CrnnVariant(input_height=32, conv_channels=(32, 64, 128, 256), lstm_hidden_size=128,
                                  lstm_layers=2, min_width=48, max_width=800),
    "crnn_base_h48": CrnnVariant(input_height=48, conv_channels=(48, 96, 192, 320), lstm_hidden_size=192,
                                 lstm_layers=2, min_width=64, max_width=1100),
}


def conv_bn_relu(in_channels: int, out_channels: int) -> list[nn.Module]:
    """3x3 conv + batch norm + ReLU."""
    return [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]


class CrnnLineRecognizer(nn.Module):
    """Grey line image (B, 1, H, W) -> CTC log-probabilities (B, W // 4, num_classes)."""

    def __init__(self, variant: CrnnVariant, num_classes: int):
        super().__init__()
        c1, c2, c3, c4 = variant.conv_channels
        layers: list[nn.Module] = []
        layers += conv_bn_relu(1, c1) + [nn.MaxPool2d(2, 2)]
        layers += conv_bn_relu(c1, c2) + [nn.MaxPool2d(2, 2)]
        layers += conv_bn_relu(c2, c3) + conv_bn_relu(c3, c3) + [nn.MaxPool2d((2, 1), (2, 1))]
        layers += conv_bn_relu(c3, c4) + conv_bn_relu(c4, c4) + [nn.MaxPool2d((2, 1), (2, 1))]
        self.features = nn.Sequential(*layers)
        remaining_height = variant.input_height // 16
        # Collapse the leftover height into channels with one full-height conv.
        self.height_collapse = nn.Sequential(nn.Conv2d(c4, c4, (remaining_height, 1), bias=False),
                                             nn.BatchNorm2d(c4), nn.ReLU(inplace=True))
        self.sequence_model = nn.LSTM(c4, variant.lstm_hidden_size, num_layers=variant.lstm_layers,
                                      bidirectional=True, batch_first=True, dropout=0.1)
        self.classifier = nn.Linear(2 * variant.lstm_hidden_size, num_classes)

    def forward(self, line_images: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) float in [-1, 1] -> (B, T, C) log-probabilities."""
        features = self.height_collapse(self.features(line_images)).squeeze(2).permute(0, 2, 1)
        sequence, _ = self.sequence_model(features)
        return self.classifier(sequence).log_softmax(dim=-1)


def timesteps_for_width(width: int) -> int:
    """Output timesteps for an input of `width` pixels (two 2x width pools)."""
    return width // WIDTH_DOWNSAMPLE_FACTOR


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
