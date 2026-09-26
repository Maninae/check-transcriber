"""Optional network harmonization of pasted checks, using PCT-Net (CNN variant).

PCT-Net (Guerreiro, Nakazawa, Stent, WACV 2023; github.com/rakutentech/PCT-Net-Image-Harmonization)
predicts a per-pixel color transform at 256x256 and applies that transform to the
full-resolution pixels. It can only recolor pixels, never repaint them, so text and
structure survive exactly, which is why it was chosen. Code and the bundled
weights are MPL-2.0. (Harmonizer, Ke et al. 2022, was the first choice but is CC BY-NC-SA 4.0.)

- The upstream repo is cloned on the data drive (`PCTNET_CODE_DIR`), not vendored, and is
  imported from there on first use. That in-body import is the optional-backend exception:
  `--no-harmonize` is the genuine fallback path.
- Runs per check on a context crop in the flat sheet plane, on albedo, before the scene light
  and the camera warp.
- The network's output is applied through harmonize_color_transfer.py: its chroma shift is
  kept (capped), its luminance shift is clamped, so white paper never darkens toward the sheet.
"""

import functools
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from scene_composer.harmonization.harmonize_color_transfer import luminance_preserving_transfer

PCTNET_CODE_DIR = Path(os.environ.get("CHECK_SYNTH_PCTNET_DIR", "/Volumes/vega/ai-models/harmonizer/pctnet"))
PCTNET_WEIGHTS_PATH = PCTNET_CODE_DIR / "pretrained_models" / "PCTNet_CNN.pth"
PCTNET_LOW_RES = 256
DEFAULT_HARMONIZE_BLEND = 0.5  # share of the network's (capped) chroma shift taken
CONTEXT_EXPANSION = 0.5  # crop margin around each check, as a fraction of its box size
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
# Copied from upstream iharm/mconfigs/base.py, entry 'CNN_pct'.
PCTNET_CNN_PARAMS = {
    "backbone_type": "ssam", "depth": 4, "ch": 32, "image_fusion": True, "attention_mid_k": 0.5,
    "batchnorm_from": 2, "attend_from": 2,
    "input_normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "dim": 3, "transform_type": "linear", "affine": True, "clamp": True, "color_space": "RGB", "use_attn": True,
}


@functools.lru_cache(maxsize=1)
def load_pctnet_cnn() -> torch.nn.Module:
    """Load PCT-Net CNN with its pretrained weights on CPU (cached per process)."""
    if not PCTNET_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"PCT-Net weights missing at {PCTNET_WEIGHTS_PATH}; see README 'Harmonization'")
    if str(PCTNET_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(PCTNET_CODE_DIR))
    from iharm.model.base.pct_net import PCTNet  # upstream code lives on the data drive, see module docstring

    model = PCTNet(**PCTNET_CNN_PARAMS)
    state = model.state_dict()
    state.update(torch.load(str(PCTNET_WEIGHTS_PATH), map_location="cpu"))
    model.load_state_dict(state)
    return model.eval()


def to_normalized_tensor(image: np.ndarray) -> torch.Tensor:
    """(H, W, 3) float [0, 1] -> (1, 3, H, W) ImageNet-normalized tensor."""
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float()
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def harmonize_region(composite: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    """Harmonize the masked foreground of one crop; returns float32 RGB [0, 1] of the same size."""
    model = load_pctnet_cnn()
    low_res_image = cv2.resize(composite, (PCTNET_LOW_RES, PCTNET_LOW_RES), interpolation=cv2.INTER_AREA)
    low_res_mask = cv2.resize(foreground_mask, (PCTNET_LOW_RES, PCTNET_LOW_RES), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        output = model(
            to_normalized_tensor(low_res_image),
            to_normalized_tensor(composite),
            torch.from_numpy(low_res_mask)[None, None].float(),
            torch.from_numpy(np.ascontiguousarray(foreground_mask))[None, None].float(),
        )
    full_res = output["images_fullres"].reshape(1, 3, *composite.shape[:2]) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(full_res[0].permute(1, 2, 0).numpy(), 0, 1).astype(np.float32)


def harmonize_pasted_checks(canvas: np.ndarray, check_masks: list[tuple[int, int, np.ndarray]],
                            blend: float = DEFAULT_HARMONIZE_BLEND) -> np.ndarray:
    """Harmonize each pasted check against its surroundings.

    Args:
        canvas: sheet-plane composite, float32 RGB [0, 1].
        check_masks: per check, (x0, y0, mask) where mask is the check's visible alpha in
            the canvas region starting at (x0, y0).
        blend: share of PCT-Net's chroma shift applied (luminance is clamped regardless).
    Returns:
        A new canvas; only pixels under the masks change (PCT-Net blends by its attention map).
    """
    harmonized = canvas.copy()
    canvas_height, canvas_width = canvas.shape[:2]
    for x0, y0, mask in check_masks:
        mask_height, mask_width = mask.shape
        margin_x, margin_y = int(mask_width * CONTEXT_EXPANSION), int(mask_height * CONTEXT_EXPANSION)
        crop_x0, crop_y0 = max(0, x0 - margin_x), max(0, y0 - margin_y)
        crop_x1, crop_y1 = min(canvas_width, x0 + mask_width + margin_x), min(canvas_height, y0 + mask_height + margin_y)
        crop_mask = np.zeros((crop_y1 - crop_y0, crop_x1 - crop_x0), np.float32)
        crop_mask[y0 - crop_y0:y0 - crop_y0 + mask_height, x0 - crop_x0:x0 - crop_x0 + mask_width] = mask
        if crop_mask.sum() < 1:
            continue
        crop = harmonized[crop_y0:crop_y1, crop_x0:crop_x1]
        result = luminance_preserving_transfer(crop, harmonize_region(crop, crop_mask), blend)
        # Keep the change strictly inside the check so the sheet itself is untouched.
        harmonized[crop_y0:crop_y1, crop_x0:crop_x1] = crop * (1 - crop_mask[..., None]) + result * crop_mask[..., None]
    return harmonized
