"""CenterNet training losses: penalty-reduced focal loss on the heatmap, scale-normalized L1 on corners.

- Heatmap: the "Objects as Points" variant of focal loss (alpha=2, beta=4). Cells
  near a peak are penalized less by (1 - target)^beta, normalized by the peak count.
- Corners: |predicted - target| offsets (cells) divided by the check diagonal (cells),
  weighted by `corner_offset_weight` (Gaussian value on supervised cells), averaged
  over the weight mass. Scale-invariant, so small checks matter as much as big ones.
"""

import torch

FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0
PROBABILITY_CLAMP = 1e-4
PEAK_TARGET_VALUE = 1.0


def center_heatmap_focal_loss(heatmap_logits: torch.Tensor, heatmap_target: torch.Tensor) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss, normalized by the number of peaks."""
    probabilities = torch.sigmoid(heatmap_logits).clamp(PROBABILITY_CLAMP, 1 - PROBABILITY_CLAMP)
    is_peak = (heatmap_target >= PEAK_TARGET_VALUE).float()
    positive_loss = torch.log(probabilities) * (1 - probabilities) ** FOCAL_ALPHA * is_peak
    negative_loss = (
        torch.log(1 - probabilities) * probabilities**FOCAL_ALPHA * (1 - heatmap_target) ** FOCAL_BETA * (1 - is_peak)
    )
    peak_count = is_peak.sum().clamp(min=1.0)
    return -(positive_loss.sum() + negative_loss.sum()) / peak_count


def corner_offset_l1_loss(
    predicted_offsets: torch.Tensor,
    offset_target: torch.Tensor,
    offset_weight: torch.Tensor,
    offset_normalizer: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean of |pred - target| / check diagonal over the 8 corner channels."""
    per_cell_error = (predicted_offsets - offset_target).abs() / offset_normalizer
    weight_mass = offset_weight.sum().clamp(min=1.0)
    return (per_cell_error * offset_weight).sum() / (weight_mass * predicted_offsets.shape[1])


def centernet_total_loss(model_outputs: dict, batch: dict, heatmap_loss_weight: float, corner_offset_loss_weight: float) -> dict[str, torch.Tensor]:
    """Both losses and their weighted sum as `total_loss`."""
    heatmap_loss = center_heatmap_focal_loss(model_outputs["center_heatmap_logits"], batch["center_heatmap_target"])
    corner_loss = corner_offset_l1_loss(
        model_outputs["corner_offsets"],
        batch["corner_offset_target"],
        batch["corner_offset_weight"],
        batch["corner_offset_normalizer"],
    )
    return {
        "total_loss": heatmap_loss_weight * heatmap_loss + corner_offset_loss_weight * corner_loss,
        "heatmap_loss": heatmap_loss,
        "corner_offset_loss": corner_loss,
    }
