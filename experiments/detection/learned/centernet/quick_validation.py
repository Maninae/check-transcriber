"""Quick validation used for checkpoint selection during training.

Runs the model on the 1280 copies of a val subset (fast to decode), maps corners to
full resolution, and scores with the shared metrics harness. The selection metric
is F1 at IoU 0.90 against the GT outline: it rewards finding every check AND
localizing it tightly, which is what the corner-precise product needs.
"""

import numpy as np
import torch

from experiments.detection.dataset.scene_annotations import SceneAnnotation
from experiments.detection.learned.centernet.centernet_inference import maps_to_detected_checks, predict_maps_for_batch
from experiments.detection.learned.centernet.check_scene_dataset import load_downscaled_scene, normalize_image_to_tensor
from experiments.detection.learned.centernet.scene_augmentation import letterbox_scene_for_evaluation
from experiments.detection.metrics.metrics_report_writing import headline_summary_line
from experiments.detection.metrics.score_predictions import score_predictions_against_split

SELECTION_IOU_KEY = "0.90"
VALIDATION_BATCH_SIZE = 4


def predict_scenes_from_downscaled_copies(
    model: torch.nn.Module, scenes: list[SceneAnnotation], input_size_pixels: int, device: str
) -> dict:
    """scene_id -> list[DetectedCheck] in full-resolution pixels."""
    predictions_by_scene_id = {}
    for batch_start in range(0, len(scenes), VALIDATION_BATCH_SIZE):
        batch_scenes = scenes[batch_start : batch_start + VALIDATION_BATCH_SIZE]
        data_dicts = [letterbox_scene_for_evaluation(load_downscaled_scene(scene), input_size_pixels) for scene in batch_scenes]
        input_batch = torch.stack([normalize_image_to_tensor(data_dict["input_image_rgb"]) for data_dict in data_dicts]).to(device)
        heatmaps, corner_offsets = predict_maps_for_batch(model, input_batch)
        for index, data_dict in enumerate(data_dicts):
            predictions_by_scene_id[data_dict["scene_id"]] = maps_to_detected_checks(
                heatmaps[index],
                corner_offsets[index],
                data_dict["source_to_input_affine"],
                1.0 / data_dict["full_resolution_to_source_scale"],
            )
    return predictions_by_scene_id


def run_quick_validation(model: torch.nn.Module, scenes: list[SceneAnnotation], input_size_pixels: int, device: str) -> tuple[float, str]:
    """(selection metric, headline line) on the given val scenes; restores train mode.

    Scene ids repeat across datasets (every set has a `val_000000`), so scenes are scored
    per dataset and the selection metric is the mean over datasets; pooling them in one
    scene-id-keyed dict would silently score scenes against another dataset's predictions.
    """
    model.eval()
    selection_metrics, headlines = [], []
    for dataset_name in sorted({scene.dataset_name for scene in scenes}):
        dataset_scenes = [scene for scene in scenes if scene.dataset_name == dataset_name]
        predictions_by_scene_id = predict_scenes_from_downscaled_copies(model, dataset_scenes, input_size_pixels, device)
        metrics = score_predictions_against_split(predictions_by_scene_id, dataset_scenes, include_records=False)
        selection_metric = metrics["detection"][SELECTION_IOU_KEY]["f1"]
        selection_metrics.append(float(np.nan_to_num(selection_metric if selection_metric is not None else 0.0)))
        headlines.append(f"{dataset_name}: {headline_summary_line(metrics)}")
    model.train()
    return float(np.mean(selection_metrics)), " || ".join(headlines)
