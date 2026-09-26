"""Constants for the learned field-mask localizer (input canvas, output stride, training defaults).

The rectified crop is resized to CANVAS_WIDTH_PX wide (aspect kept) and pasted top-left on a
CANVAS_WIDTH_PX x CANVAS_HEIGHT_PX zero canvas; every size kind (personal 0.458, money order 0.446,
business 0.412 height/width) fits, so the ONNX model has one fixed input shape.
"""

from experiments.field_reading.config import TARGET_FIELD_NAMES

ENCODER_NAME = "mobilenetv3_large_100.ra_in1k"   # timm, Apache-2.0 weights (ImageNet-1k)
CANVAS_WIDTH_PX = 768
CANVAS_HEIGHT_PX = 352
OUTPUT_STRIDE = 2
DECODER_CHANNELS = 64
FIELD_CHANNEL_NAMES: list[str] = list(TARGET_FIELD_NAMES)
IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB = (0.229, 0.224, 0.225)
SEGNET_METHOD_ID = f"segnet_mobilenetv3l_{CANVAS_WIDTH_PX}"
