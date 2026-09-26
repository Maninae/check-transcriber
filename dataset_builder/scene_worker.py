"""Generate and write one dataset scene; runs inside a worker process.

The pixels and label come from `scene_composer.on_demand.compose_scene_on_demand`, the one
composition path (a `SyntheticSceneStream` yields the same scenes); this module only adds files.

- Each scene is addressed by (seed, split, scene_index), so any single scene can be regenerated
  without rebuilding the others.
- Checks and the background come only from the task's split pools (contract C4).
- The scene annotation is written last, via rename, so its existence means the scene is complete
  (resume in `scene_task_runner.py` relies on this).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from dataset_builder.exports.annotation_exports import write_yolo_labels
from scene_composer.on_demand import compose_scene_on_demand, scene_id_for
from scene_composer.scene_config import SceneConfig
from scene_composer.scene_ingredient_pools import SceneIngredientPools


@dataclass(frozen=True)
class SceneTask:
    """Everything a worker needs to produce one scene."""

    seed: int
    split_name: str
    scene_index: int
    pools: SceneIngredientPools
    split_directory: str
    template_count: int
    scene_config: SceneConfig


def scene_id_for_task(task: SceneTask) -> str:
    """Stable scene id, e.g. `val_000012`; also the stem of every per-scene file."""
    return scene_id_for(task.split_name, task.scene_index)


def worker_initializer(harmonize: bool) -> None:
    """One thread per library per worker; the pool provides the parallelism."""
    cv2.setNumThreads(1)
    if harmonize:
        import torch  # optional backend, only loaded when --harmonize is on

        torch.set_num_threads(1)


def generate_scene(task: SceneTask) -> dict:
    """Compose one scene and write its photo, YOLO labels and annotation. Returns a summary for progress and the manifest."""
    composed = compose_scene_on_demand(task.seed, task.split_name, task.scene_index, pools=task.pools,
                                       template_count=task.template_count, scene_config=task.scene_config)
    label_dict = composed.label_record()
    split_directory = Path(task.split_directory)
    composed.write_jpeg(split_directory / "images" / composed.label.image_file)
    write_yolo_labels(label_dict, split_directory)
    annotation_path = split_directory / "annotations" / f"{composed.label.scene_id}.json"
    partial_path = annotation_path.with_suffix(".json.partial")
    partial_path.write_text(json.dumps(label_dict))
    partial_path.replace(annotation_path)
    return {"split": task.split_name, "scene_id": composed.label.scene_id, "check_count": len(composed.label.checks),
            "template_ids": sorted({check.template_id for check in composed.label.checks}),
            "background_id": composed.label.background_id}
