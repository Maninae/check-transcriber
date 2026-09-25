"""Generate and write one scene; runs inside a worker process.

Each scene's randomness comes from `np.random.default_rng([seed, split_index, scene_index])`,
so any single scene can be regenerated without rebuilding the others.
"""

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from synth.backgrounds.loader import load_background_rgb
from synth.compose.compose_scene import SceneConfig, compose_scene, sample_check_count
from synth.dataset.annotation_exports import write_yolo_labels
from synth.dataset.splits import SPLIT_NAMES
from synth.render.check_templates import TemplateDesign, build_template_catalog
from synth.render.fake_data import sample_check_content
from synth.render.render_check import render_check

OUTPUT_JPEG_QUALITY = 95  # the scene already went through a phone-quality JPEG; this save must not degrade it further
BACKGROUND_CACHE_SIZE = 3


@dataclass(frozen=True)
class SceneTask:
    """Everything a worker needs to produce one scene."""

    seed: int
    split_name: str
    scene_index: int
    template_ids: tuple[str, ...]
    backgrounds: tuple[tuple[str, str], ...]  # (background_id, file path)
    split_directory: str
    template_count: int
    scene_config: SceneConfig


@functools.lru_cache(maxsize=4)
def template_catalog_by_id(template_count: int) -> dict[str, TemplateDesign]:
    """Template designs keyed by id (cached per process)."""
    return {template.template_id: template for template in build_template_catalog(template_count)}


@functools.lru_cache(maxsize=BACKGROUND_CACHE_SIZE)
def cached_background(file_path: str) -> np.ndarray:
    """Decoded background, cached because consecutive scenes often share one."""
    return load_background_rgb(Path(file_path))


def worker_initializer(harmonize: bool) -> None:
    """One thread per library per worker; the pool provides the parallelism."""
    cv2.setNumThreads(1)
    if harmonize:
        import torch  # optional backend, only loaded when --harmonize is on

        torch.set_num_threads(1)


def generate_scene(task: SceneTask) -> dict:
    """Render, compose, and write one scene. Returns a small summary for progress and manifest counts."""
    rng = np.random.default_rng([task.seed, SPLIT_NAMES.index(task.split_name), task.scene_index])
    catalog = template_catalog_by_id(task.template_count)
    check_count = sample_check_count(rng)
    rendered_checks = []
    for _ in range(check_count):
        template = catalog[task.template_ids[int(rng.integers(len(task.template_ids)))]]
        rendered_checks.append(render_check(template, sample_check_content(template, rng), rng))
    background_id, background_path = task.backgrounds[int(rng.integers(len(task.backgrounds)))]
    scene_id = f"{task.split_name}_{task.scene_index:06d}"
    photo, label = compose_scene(scene_id, rendered_checks, cached_background(background_path), background_id, rng, task.scene_config)

    split_directory = Path(task.split_directory)
    cv2.imwrite(str(split_directory / "images" / label.image_file), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY])
    label_dict = label.to_dict()
    (split_directory / "annotations" / f"{scene_id}.json").write_text(json.dumps(label_dict))
    write_yolo_labels(label_dict, split_directory)
    return {"split": task.split_name, "scene_id": scene_id, "check_count": check_count,
            "template_ids": sorted({check.template_id for check in label.checks}), "background_id": background_id}
