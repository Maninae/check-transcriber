"""What kind of surface each background is, and how scenes choose among a split's backgrounds.

Two traits per background id:
- `is_lit_photo`: a real (or FLUX-photographic) picture with wrinkles, folds and light falloff,
  as opposed to an evenly lit tileable material swatch (ambientCG, Poly Haven).
- `is_soft`: cloth-like (bedsheet, blanket, fabric, rug, leather) rather than a hard table or floor.

Sources of truth, by accepted subfolder:
- `web/`: the `source` and `category` columns of `web/SOURCES.jsonl` (Commons / Openverse are photos).
- `flux/`: the surface slug in the filename (`bg_0007__blue-bedsheet__seed=...png`); all are photographic.
- `photos/`: Owen's own pictures of bedsheets: lit and soft.
- anything else (procedural test fabrics): flat and soft.

Choosing (`choose_background`): lit photos get `LIT_PHOTO_SCENE_SHARE` of a split's scenes whenever
the split has any, and soft surfaces are `SOFT_SURFACE_WEIGHT` times as likely within each group, so
the scene mix leans toward the surfaces checks are actually photographed on. Split membership is untouched:
the choice only ever draws from the split's own pool.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LIT_PHOTO_SCENE_SHARE = 0.6
SOFT_SURFACE_WEIGHT = 2.0
WEB_SOURCES_FILE = "SOURCES.jsonl"
LIT_WEB_SOURCES = frozenset({"commons", "openverse"})
SOFT_WEB_CATEGORIES = frozenset({"bedding", "fabric", "rug_carpet", "leather"})
SOFT_FLUX_SLUG_WORDS = ("bedsheet", "duvet", "comforter", "blanket", "throw", "couch", "cushion", "carpet", "rug",
                        "tablecloth", "linen")


@dataclass(frozen=True)
class SceneBackground:
    """One background a scene may use: id, file path, and its surface traits."""

    background_id: str
    file_path: str
    is_lit_photo: bool
    is_soft: bool


def load_web_source_rows(background_root: Path) -> dict[str, dict]:
    """`web/SOURCES.jsonl` rows keyed by background id (`web/<file>`); empty when the file is absent."""
    sources_path = background_root / "web" / WEB_SOURCES_FILE
    if not sources_path.exists():
        return {}
    rows = [json.loads(line) for line in sources_path.read_text().splitlines() if line.strip()]
    return {f"web/{row['file']}": row for row in rows}


def surface_traits(background_id: str, web_rows: dict[str, dict]) -> tuple[bool, bool]:
    """(is_lit_photo, is_soft) for one background id; see the module docstring for the rules."""
    folder = background_id.split("/", 1)[0]
    if folder == "web" and background_id in web_rows:
        row = web_rows[background_id]
        return row["source"] in LIT_WEB_SOURCES, row["category"] in SOFT_WEB_CATEGORIES
    if folder == "flux":
        return True, any(word in background_id for word in SOFT_FLUX_SLUG_WORDS)
    if folder == "photos":
        return True, True
    return False, True


def describe_backgrounds(background_root: Path, backgrounds: list[tuple[str, str]]) -> tuple[SceneBackground, ...]:
    """Attach traits to (background_id, file path) pairs, keeping their order."""
    web_rows = load_web_source_rows(background_root)
    return tuple(SceneBackground(background_id, file_path, *surface_traits(background_id, web_rows))
                 for background_id, file_path in backgrounds)


def choose_background(backgrounds: tuple[SceneBackground, ...], rng: np.random.Generator) -> SceneBackground:
    """Pick a lit photo `LIT_PHOTO_SCENE_SHARE` of the time (when any exist), soft surfaces weighted up."""
    lit = [background for background in backgrounds if background.is_lit_photo]
    flat = [background for background in backgrounds if not background.is_lit_photo]
    use_lit = bool(lit) and (not flat or rng.random() < LIT_PHOTO_SCENE_SHARE)
    group = lit if use_lit else flat
    weights = np.array([SOFT_SURFACE_WEIGHT if background.is_soft else 1.0 for background in group])
    return group[int(rng.choice(len(group), p=weights / weights.sum()))]
