"""Generate and write one scene; runs inside a worker process.

Each scene's randomness comes from `np.random.default_rng([seed, split_index, scene_index])`,
so any single scene can be regenerated without rebuilding the others.

- Checks draw templates, fonts, payees, banks and the background only from the task's split pools (contract C4).
- The background is chosen by `choose_background` (lit photos favoured, soft surfaces weighted up);
  flat soft swatches get procedural cloth relief before compositing.
- The scene annotation is written last, via rename, so its existence means the scene is complete
  (resume in `scene_task_runner.py` relies on this).
"""

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dataset_builder.exports.annotation_exports import write_yolo_labels
from scene_composer.compose_scene import SceneConfig, compose_scene, sample_check_count
from synthetic_backgrounds.background_traits import SceneBackground, choose_background
from synthetic_backgrounds.loader import load_background_rgb
from synthetic_backgrounds.surface_relief import add_cloth_relief
from synthetic_checks.check_templates import TemplateDesign, build_template_catalog
from synthetic_checks.content.fake_data import sample_check_content
from synthetic_checks.render_check import render_check
from synthetic_checks.splits import SPLIT_NAMES

OUTPUT_JPEG_QUALITY = 95  # the scene already went through a phone-quality JPEG; this save must not degrade it further
BACKGROUND_CACHE_SIZE = 3


@dataclass(frozen=True)
class SceneTask:
    """Everything a worker needs to produce one scene."""

    seed: int
    split_name: str
    scene_index: int
    template_ids: tuple[str, ...]
    backgrounds: tuple[SceneBackground, ...]  # the split's backgrounds with their surface traits
    split_directory: str
    template_count: int
    scene_config: SceneConfig
    handwriting_font_ids: tuple[str, ...]
    signature_font_ids: tuple[str, ...]
    payee_names: tuple[str, ...]
    bank_names: tuple[str, ...]


def scene_id_for_task(task: SceneTask) -> str:
    """Stable scene id, e.g. `val_000012`; also the stem of every per-scene file."""
    return f"{task.split_name}_{task.scene_index:06d}"


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
        rendered_checks.append(render_check(template, sample_check_content(template, rng, payee_names=task.payee_names,
                                                                         bank_names=task.bank_names), rng,
                                            handwriting_font_ids=list(task.handwriting_font_ids),
                                            signature_font_ids=list(task.signature_font_ids)))
    background = choose_background(task.backgrounds, rng)
    background_rgb = cached_background(background.file_path)
    if background.is_soft and not background.is_lit_photo:
        background_rgb = add_cloth_relief(background_rgb, rng)  # a flat swatch becomes a slept-on sheet
    scene_id = scene_id_for_task(task)
    photo, label = compose_scene(scene_id, rendered_checks, background_rgb, background.background_id, rng, task.scene_config)

    split_directory = Path(task.split_directory)
    cv2.imwrite(str(split_directory / "images" / label.image_file), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY])
    label_dict = label.to_dict()
    write_yolo_labels(label_dict, split_directory)
    annotation_path = split_directory / "annotations" / f"{scene_id}.json"
    partial_path = annotation_path.with_suffix(".json.partial")
    partial_path.write_text(json.dumps(label_dict))
    partial_path.replace(annotation_path)
    return {"split": task.split_name, "scene_id": scene_id, "check_count": check_count,
            "template_ids": sorted({check.template_id for check in label.checks}), "background_id": background.background_id}
