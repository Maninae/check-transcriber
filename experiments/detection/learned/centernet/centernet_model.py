"""CenterNet check detector: MobileNetV3 backbone, FPN-style neck to stride 4, two heads.

Licensing: torchvision's MobileNetV3 code is BSD-3-Clause. The optional ImageNet-1k
weights (`IMAGENET1K_V1`) were trained by the torchvision team on ImageNet; torchvision
publishes them without an extra license, but ImageNet itself is research-only, so the
provenance is noted here for whoever ships the model. No AGPL code or weights anywhere.

Outputs (raw, before sigmoid) at stride `OUTPUT_STRIDE`:
- `center_heatmap_logits` (B, 1, G, G): one check center per peak.
- `corner_offsets` (B, 8, G, G): TL,TR,BR,BL (dx, dy) from the cell center, in cells
  (head output times `CORNER_OFFSET_OUTPUT_SCALE_CELLS`, a constant Mul in the graph).

ONNX notes: only Conv/BN/ReLU/Hardswish/HardSigmoid/nearest-Resize/Add/Mul, so the graph
runs on onnxruntime-web's WASM backend; `CenterNetExportWrapper` bakes in the sigmoid.
"""

import torch
import torchvision
from torch import nn
from torch.nn import functional

from experiments.detection.learned.centernet.centernet_config import (
    CORNER_OFFSET_CHANNELS,
    CORNER_OFFSET_OUTPUT_SCALE_CELLS,
    PRETRAINED_WEIGHTS_DIRECTORY,
)

# Index of the last `features` block at stride 4, 8, 16, 32.
BACKBONE_TAP_INDICES = {
    "mobilenet_v3_large": (3, 6, 12, 16),
    "mobilenet_v3_small": (1, 3, 8, 12),
}
BACKBONE_WEIGHTS = {
    "mobilenet_v3_large": torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    "mobilenet_v3_small": torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
}
HEAD_HIDDEN_CHANNELS = 64
HEATMAP_PRIOR_PROBABILITY = 0.1  # initial sigmoid output; bias = -log((1-p)/p) as in CenterNet/focal loss
SHAPE_PROBE_SIZE = 64


def conv_bn_relu(input_channels: int, output_channels: int, kernel_size: int) -> nn.Sequential:
    """Conv + BatchNorm + ReLU with 'same' padding."""
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )


class TopDownFeatureNeck(nn.Module):
    """FPN top-down path: 1x1 laterals, nearest 2x upsample + add, 3x3 smoothing; ends at stride 4."""

    def __init__(self, backbone_channels: list[int], neck_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList(conv_bn_relu(channels, neck_channels, 1) for channels in backbone_channels)
        self.smoothing_convs = nn.ModuleList(conv_bn_relu(neck_channels, neck_channels, 3) for _ in backbone_channels[:-1])

    def forward(self, features_fine_to_coarse: list[torch.Tensor]) -> torch.Tensor:
        """Merge stride-4..32 features into one stride-4 map."""
        merged = self.lateral_convs[-1](features_fine_to_coarse[-1])
        for level in range(len(features_fine_to_coarse) - 2, -1, -1):
            upsampled = functional.interpolate(merged, scale_factor=2.0, mode="nearest")
            merged = self.smoothing_convs[level](self.lateral_convs[level](features_fine_to_coarse[level]) + upsampled)
        return merged


def prediction_head(input_channels: int, output_channels: int) -> nn.Sequential:
    """3x3 conv + ReLU + 1x1 projection."""
    return nn.Sequential(
        nn.Conv2d(input_channels, HEAD_HIDDEN_CHANNELS, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(HEAD_HIDDEN_CHANNELS, output_channels, 1),
    )


class CenterNetCheckDetector(nn.Module):
    """Backbone + neck + heatmap/corner-offset heads (see module docstring)."""

    def __init__(self, backbone_name: str = "mobilenet_v3_large", neck_channels: int = 96, pretrained_backbone: bool = True):
        super().__init__()
        backbone_weights = None
        if pretrained_backbone:
            PRETRAINED_WEIGHTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(PRETRAINED_WEIGHTS_DIRECTORY))  # keep the download on vega
            backbone_weights = BACKBONE_WEIGHTS[backbone_name]
        self.backbone_features = getattr(torchvision.models, backbone_name)(weights=backbone_weights).features
        self.tap_indices = BACKBONE_TAP_INDICES[backbone_name]
        del self.backbone_features[self.tap_indices[-1] + 1 :]
        with torch.no_grad():
            probe_features = self.extract_backbone_features(torch.zeros(1, 3, SHAPE_PROBE_SIZE, SHAPE_PROBE_SIZE))
        self.neck = TopDownFeatureNeck([feature.shape[1] for feature in probe_features], neck_channels)
        self.center_heatmap_head = prediction_head(neck_channels, 1)
        self.corner_offset_head = prediction_head(neck_channels, CORNER_OFFSET_CHANNELS)
        heatmap_bias = -torch.log(torch.tensor((1 - HEATMAP_PRIOR_PROBABILITY) / HEATMAP_PRIOR_PROBABILITY))
        nn.init.constant_(self.center_heatmap_head[-1].bias, float(heatmap_bias))

    def extract_backbone_features(self, input_image: torch.Tensor) -> list[torch.Tensor]:
        """Backbone activations at strides 4, 8, 16, 32."""
        features = []
        activations = input_image
        for block_index, block in enumerate(self.backbone_features):
            activations = block(activations)
            if block_index in self.tap_indices:
                features.append(activations)
        return features

    def forward(self, input_image: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, 3, S, S) normalized RGB -> raw head outputs at stride 4."""
        neck_features = self.neck(self.extract_backbone_features(input_image))
        return {
            "center_heatmap_logits": self.center_heatmap_head(neck_features),
            "corner_offsets": self.corner_offset_head(neck_features) * CORNER_OFFSET_OUTPUT_SCALE_CELLS,
        }


class CenterNetExportWrapper(nn.Module):
    """ONNX-facing wrapper: returns (sigmoid heatmap, corner offsets) as a plain tuple."""

    def __init__(self, detector: CenterNetCheckDetector):
        super().__init__()
        self.detector = detector

    def forward(self, input_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Network outputs ready for `centernet_decoding`."""
        outputs = self.detector(input_image)
        return torch.sigmoid(outputs["center_heatmap_logits"]), outputs["corner_offsets"]
