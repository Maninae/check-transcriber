"""Background choice (lit photos favoured, soft weighted up, split-only) and cloth relief on flat swatches."""

import json

import numpy as np

from synthetic_backgrounds.background_traits import (
    LIT_PHOTO_SCENE_SHARE,
    SceneBackground,
    choose_background,
    describe_backgrounds,
)
from synthetic_backgrounds.surface_relief import add_cloth_relief


def test_traits_come_from_sources_and_filenames(tmp_path):
    """Commons photos are lit, ambientCG swatches flat; FLUX bedsheets soft, FLUX laminate hard."""
    (tmp_path / "web").mkdir()
    rows = [{"file": "a.jpg", "source": "commons", "category": "bedding"},
            {"file": "b.jpg", "source": "ambientcg", "category": "wood_table"}]
    (tmp_path / "web" / "SOURCES.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    described = describe_backgrounds(tmp_path, [("web/a.jpg", "x"), ("web/b.jpg", "y"),
                                                ("flux/bg_0001__white-bedsheet__seed=1.png", "z"),
                                                ("flux/bg_0003__desk-laminate__seed=3.png", "w")])
    assert [(b.is_lit_photo, b.is_soft) for b in described] == [(True, True), (False, False), (True, True), (True, False)]


def test_lit_photos_get_their_share_and_choice_stays_in_pool():
    """Over many scenes lit photos land near LIT_PHOTO_SCENE_SHARE, and only pool ids are ever chosen."""
    pool = (SceneBackground("flux/a", "a", True, True),) + tuple(SceneBackground(f"web/t{i}", "t", False, i % 2 == 0)
                                                                  for i in range(20))
    rng = np.random.default_rng(0)
    chosen = [choose_background(pool, rng) for _ in range(4000)]
    lit_share = sum(b.is_lit_photo for b in chosen) / len(chosen)
    assert abs(lit_share - LIT_PHOTO_SCENE_SHARE) < 0.03
    assert {b.background_id for b in chosen} <= {b.background_id for b in pool}


def test_cloth_relief_shades_a_flat_swatch_without_changing_its_level():
    """A uniform swatch gains visible low-frequency variation; mean brightness stays close; deterministic."""
    swatch = np.full((600, 800, 3), 180, np.uint8)
    shaded = add_cloth_relief(swatch, np.random.default_rng(4))
    assert shaded.shape == swatch.shape and shaded.dtype == np.uint8
    assert shaded.std() > 5
    assert abs(float(shaded.mean()) - 180) < 12
    assert np.array_equal(shaded, add_cloth_relief(swatch, np.random.default_rng(4)))
