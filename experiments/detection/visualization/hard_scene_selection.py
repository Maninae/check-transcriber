"""Pick one representative hard scene per difficulty from a split, deterministically.

Each criterion names a difficulty Owen asked to see (rug, overlap, shadow, upside
down, out of frame, ...) and scores scenes; the highest-scoring scene not already
picked wins. Selection reads only the annotations, never detector results, so the
showcase is not cherry-picked for any detector.
"""

from collections.abc import Callable

from experiments.detection.dataset.scene_annotations import SceneAnnotation

HardSceneCriterion = tuple[str, Callable[[SceneAnnotation], float]]

HARD_SCENE_CRITERIA: list[HardSceneCriterion] = [
    ("busy-rug-or-pattern", lambda scene: len(scene.checks) if scene.background_category in ("carpet_rug", "fabric_pattern") else -1),
    ("heavy-overlap", lambda scene: 1 - min(check.visible_fraction for check in scene.checks)),
    ("hand-and-phone-shadow", lambda scene: len(scene.checks) if scene.cast_shadow_kind == "phone_and_hand" else -1),
    ("upside-down-and-sideways", lambda scene: sum(check.orientation_class != 0 for check in scene.checks)),
    ("out-of-frame", lambda scene: 1 - min(check.in_frame_fraction for check in scene.checks)),
    ("white-bedding", lambda scene: len(scene.checks) if scene.background_category == "bedding" else -1),
    ("folded-and-curled", lambda scene: sum("fold" in check.deformation_kinds and "curl" in check.deformation_kinds for check in scene.checks)),
    ("twelve-checks-loose", lambda scene: len(scene.checks) + (0.5 if scene.layout_mode != "grid" else 0)),
]


def select_hard_scenes(scenes: list[SceneAnnotation]) -> list[tuple[str, SceneAnnotation]]:
    """Return (criterion name, scene) pairs, one distinct scene per criterion."""
    picked_scene_ids: set[str] = set()
    selections = []
    for criterion_name, score_scene in HARD_SCENE_CRITERIA:
        candidates = sorted(
            (scene for scene in scenes if scene.scene_id not in picked_scene_ids),
            key=lambda scene: (-score_scene(scene), scene.scene_id),
        )
        chosen_scene = candidates[0]
        picked_scene_ids.add(chosen_scene.scene_id)
        selections.append((criterion_name, chosen_scene))
    return selections
