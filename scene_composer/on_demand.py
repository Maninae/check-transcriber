"""Compose a scene on demand: seed in, phone-photo image and full label out, nothing written to disk.

This is the one composition path. `compose_scene_on_demand` draws the checks (templates, content,
fonts, payees, banks), a background and every effect from `default_rng([seed, split_index, scene_index])`,
so a scene is addressable by (seed, split, scene_index) alone:

- `SyntheticSceneStream` yields scene 0, 1, 2, ... of one split lazily (a training loop, a demo).
- `dataset_builder/scene_worker.py` calls the same function and only adds the file writes, so
  `SyntheticSceneStream(seed, split)` scene i is the dataset builder's `<split>_<i:06d>` for the same
  seed, background root and template count (`test_stream_reproduces_the_dataset_builders_scenes`).
- `generate_one.py` is the CLI over a single scene.

Split pools (contract C4) are honoured: scenes draw only from their split's `SceneIngredientPools`.
"""

import functools
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scene_composer import GENERATOR_VERSION
from scene_composer.compose_scene import compose_scene, sample_check_count
from scene_composer.scene_config import SceneConfig
from scene_composer.scene_ingredient_pools import SceneIngredientPools, library_ingredient_pools, require_known_split
from scene_composer.scene_label import SceneLabel
from synthetic_backgrounds.background_traits import choose_background
from synthetic_backgrounds.loader import load_background_rgb
from synthetic_backgrounds.surface_relief import add_cloth_relief
from synthetic_checks.check_templates import DEFAULT_TEMPLATE_COUNT, TemplateDesign, build_template_catalog
from synthetic_checks.content.fake_data import sample_check_content
from synthetic_checks.render_check import render_check
from synthetic_checks.splits import SPLIT_NAMES
from synthetic_data_paths import BACKGROUND_DIR

SCENE_FILE_JPEG_QUALITY = 95  # the photo already went through a phone-quality JPEG; saving must not degrade it further
BACKGROUND_CACHE_SIZE = 3     # consecutive scenes often share a background; decoding one costs ~0.1 s


@dataclass(frozen=True)
class ComposedScene:
    """One finished scene: the photo, its label, and where it came from."""

    seed: int
    split: str
    scene_index: int
    photo_rgb: np.ndarray   # (H, W, 3) uint8 RGB, already through the simulated phone JPEG
    label: SceneLabel       # corners, outline, orientation, field quads, visibility, effects (contract C3)

    def label_record(self) -> dict:
        """The full label as plain JSON types; identical to a dataset annotation file."""
        return self.label.to_dict()

    def provenance(self) -> dict:
        """What regenerates this exact scene (plus the generator version that produced it)."""
        return {"generator_version": GENERATOR_VERSION, "seed": self.seed, "split": self.split, "scene_index": self.scene_index}

    def photo_image(self) -> Image.Image:
        """The photo as a PIL image."""
        return Image.fromarray(self.photo_rgb)

    def write_jpeg(self, output_path: Path) -> None:
        """Save the photo the way every dataset scene is saved (quality `SCENE_FILE_JPEG_QUALITY`)."""
        if not cv2.imwrite(str(output_path), cv2.cvtColor(self.photo_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, SCENE_FILE_JPEG_QUALITY]):
            raise OSError(f"could not write {output_path}")


def scene_id_for(split: str, scene_index: int) -> str:
    """Stable scene id, e.g. `val_000012`; also the stem of every per-scene dataset file."""
    return f"{split}_{scene_index:06d}"


@functools.lru_cache(maxsize=4)
def template_catalog_by_id(template_count: int) -> dict[str, TemplateDesign]:
    """Template designs keyed by id (cached per process)."""
    return {template.template_id: template for template in build_template_catalog(template_count)}


@functools.lru_cache(maxsize=BACKGROUND_CACHE_SIZE)
def cached_background(file_path: str) -> np.ndarray:
    """Decoded background, cached because consecutive scenes often share one."""
    return load_background_rgb(Path(file_path))


def resolve_scene_config(scene_config: SceneConfig | None, harmonize: bool) -> SceneConfig:
    """`harmonize=True` is shorthand for `SceneConfig(harmonize=True)`; passing both is ambiguous."""
    if scene_config is None:
        return SceneConfig(harmonize=harmonize)
    if harmonize and not scene_config.harmonize:
        raise ValueError("pass harmonize inside scene_config, not alongside it")
    return scene_config


def compose_scene_on_demand(seed: int, split: str = "train", scene_index: int = 0, *,
                            pools: SceneIngredientPools | None = None, backgrounds: Path = BACKGROUND_DIR,
                            template_count: int = DEFAULT_TEMPLATE_COUNT, harmonize: bool = False,
                            scene_config: SceneConfig | None = None) -> ComposedScene:
    """Render and compose one scene in memory. Same arguments, same pixels and label, every time.

    Args:
        seed: split seed and scene seed at once (as `build_dataset --seed`).
        split: `train`, `val` or `eval`; the scene uses only that split's ids.
        scene_index: which scene of the split; the rng is `default_rng([seed, split_index, scene_index])`.
        pools: the split's ingredients; default plans this seed's split of everything under `backgrounds`.
        backgrounds: background root (only its accepted subfolders count); ignored when `pools` is given.
        template_count: size of the template catalog the template ids come from.
        harmonize: shorthand for `SceneConfig(harmonize=True)` (needs torch + PCT-Net).
        scene_config: every composition knob; default `SceneConfig()`.
    """
    require_known_split(split)
    config = resolve_scene_config(scene_config, harmonize)
    if pools is None:
        pools = library_ingredient_pools(seed, split, backgrounds, template_count)
    rng = np.random.default_rng([seed, SPLIT_NAMES.index(split), scene_index])
    catalog = template_catalog_by_id(template_count)
    rendered_checks = []
    for _ in range(sample_check_count(rng)):
        template = catalog[pools.template_ids[int(rng.integers(len(pools.template_ids)))]]
        content = sample_check_content(template, rng, payee_names=pools.payee_names, bank_names=pools.bank_names)
        rendered_checks.append(render_check(template, content, rng, handwriting_font_ids=list(pools.handwriting_font_ids),
                                            signature_font_ids=list(pools.signature_font_ids)))
    background = choose_background(pools.backgrounds, rng)
    background_rgb = cached_background(background.file_path)
    if background.is_soft and not background.is_lit_photo:
        background_rgb = add_cloth_relief(background_rgb, rng)  # a flat swatch becomes a slept-on sheet
    photo, label = compose_scene(scene_id_for(split, scene_index), rendered_checks, background_rgb,
                                 background.background_id, rng, config)
    return ComposedScene(seed=seed, split=split, scene_index=scene_index, photo_rgb=photo, label=label)


class SyntheticSceneStream:
    """Lazy scenes of one split, scene after scene, each composed only when asked for.

    Iterating yields scenes `start_index`, `start_index + 1`, ... (endless unless `scene_count` is set);
    iterating again starts over and yields the same scenes. `scene(i)` gives random access.
    The split's pools are planned once, at construction, exactly as the dataset builder plans them.
    """

    def __init__(self, seed: int, split: str = "train", backgrounds: Path = BACKGROUND_DIR, harmonize: bool = False, *,
                 template_count: int = DEFAULT_TEMPLATE_COUNT, scene_config: SceneConfig | None = None,
                 start_index: int = 0, scene_count: int | None = None):
        require_known_split(split)
        self.seed, self.split = seed, split
        self.template_count = template_count
        self.scene_config = resolve_scene_config(scene_config, harmonize)
        self.start_index, self.scene_count = start_index, scene_count
        self.pools = library_ingredient_pools(seed, split, Path(backgrounds), template_count)

    def scene(self, scene_index: int) -> ComposedScene:
        """Scene `scene_index` of this split (independent of what was generated before)."""
        return compose_scene_on_demand(self.seed, self.split, scene_index, pools=self.pools,
                                       template_count=self.template_count, scene_config=self.scene_config)

    def __iter__(self) -> Iterator[ComposedScene]:
        scene_index = self.start_index
        while self.scene_count is None or scene_index < self.start_index + self.scene_count:
            yield self.scene(scene_index)
            scene_index += 1
