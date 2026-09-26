"""Turn an RGB field crop into the CRNN's input: grey, fixed height, bounded width, [-1, 1].

This is the exact recipe the browser must reproduce before calling the ONNX model:
luma grey (0.299 R + 0.587 G + 0.114 B), bilinear resize to `input_height` keeping aspect,
width clamped to [min_width, max_width] (squashed if wider), then (x / 255 - 0.5) / 0.5.
Batches are right-padded with PAD_VALUE; the valid width sets the CTC timesteps.
On MPS, pad the batch width up to a multiple of MPS_WIDTH_MULTIPLE: MPSGraph compiles new Metal
kernels for every distinct input shape, so free-varying widths make each step a shader compile.
"""

import cv2
import numpy as np

PAD_VALUE = 1.0   # padded pixels look like white paper after normalization
MPS_WIDTH_MULTIPLE = 128


def preprocess_line_image(rgb_image: np.ndarray, input_height: int, min_width: int, max_width: int) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (input_height, W') float32 in [-1, 1]."""
    grey = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) if rgb_image.ndim == 3 else rgb_image
    height, width = grey.shape
    target_width = int(round(width * input_height / max(1, height)))
    target_width = int(np.clip(target_width, min_width, max_width))
    resized = cv2.resize(grey, (target_width, input_height), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0 - 0.5) / 0.5


def pad_line_batch(line_images: list[np.ndarray], width_multiple: int = 1) -> tuple[np.ndarray, list[int]]:
    """Stack lines into (B, 1, H, W) right-padded to the max width rounded up to `width_multiple`.

    Returns the batch and each line's valid (unpadded) width.
    """
    widths = [image.shape[1] for image in line_images]
    batch_width = -(-max(widths) // width_multiple) * width_multiple
    batch = np.full((len(line_images), 1, line_images[0].shape[0], batch_width), PAD_VALUE, dtype=np.float32)
    for index, image in enumerate(line_images):
        batch[index, 0, :, :image.shape[1]] = image
    return batch, widths
