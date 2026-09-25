"""Handwriting engine invariants: font pool, glyph coverage, exact ink boxes, variety, speed."""

import time
import warnings

import cv2
import numpy as np
import pytest
from PIL import Image

from synth.paths import FONT_DIR
from synth.render.fonts import APACHE_LICENSE, FONT_SPECS, OFL_LICENSE, FontRole, font_ids_with_role
from synth.render.handwriting_font_metrics import missing_characters
from synth.render.handwriting_writer import draw_handwritten_field, sample_writer
from synth.render.pen_centreline import centreline_endpoints, prune_centreline_spurs, thin_to_centreline
from synth.render.text_drawing import ALPHA_INK_THRESHOLD

PEN_ROLES = (FontRole.HANDWRITING, FontRole.SIGNATURE)
PEN_FONT_SPECS = [spec for spec in FONT_SPECS if spec.role in PEN_ROLES]
BLUE_INK = (20, 30, 90)
BLACK_INK = (25, 25, 30)
FILL_IN_EM_PX = 54
SIGNATURE_EM_PX = 90


def transparent_canvas(width: int = 1400, height: int = 200) -> Image.Image:
    """RGBA canvas with zero alpha: after compositing, its alpha IS the ink alpha."""
    return Image.new("RGBA", (width, height), (255, 255, 255, 0))


def ink_bbox(canvas: Image.Image) -> tuple[int, int, int, int] | None:
    """Tight (x0, y0, x1, y1) box of canvas pixels whose alpha counts as ink."""
    alpha = np.asarray(canvas)[..., 3]
    rows = np.flatnonzero((alpha > ALPHA_INK_THRESHOLD).any(axis=1))
    if len(rows) == 0:
        return None
    columns = np.flatnonzero((alpha > ALPHA_INK_THRESHOLD).any(axis=0))
    return int(columns[0]), int(rows[0]), int(columns[-1] + 1), int(rows[-1] + 1)


def test_font_pool_is_large_and_open_licensed():
    assert len(font_ids_with_role(FontRole.HANDWRITING)) >= 24
    assert len(font_ids_with_role(FontRole.SIGNATURE)) >= 8
    for spec in PEN_FONT_SPECS:
        assert spec.license_name in (OFL_LICENSE, APACHE_LICENSE), spec.font_id
        assert "/ofl/" in spec.source_url or "/apache/" in spec.source_url, spec.font_id


@pytest.mark.parametrize("spec", PEN_FONT_SPECS, ids=lambda spec: spec.font_id)
def test_pen_font_covers_every_character_we_write(spec):
    font_path = FONT_DIR / spec.relative_path
    if not font_path.exists():
        warnings.warn(f"font {spec.font_id} not fetched ({font_path}); run python -m synth.render.fetch_fonts")
        pytest.skip(f"MISSING FONT FILE {font_path}")
    assert missing_characters(spec.font_id) == []


def test_missing_glyph_detector_flags_characters_a_latin_font_lacks():
    assert missing_characters("kalam", "漢字") == ["漢", "字"]


@pytest.mark.parametrize("seed", range(6))
def test_returned_box_is_exactly_the_laid_down_ink(seed):
    rng = np.random.default_rng(seed)
    writer = sample_writer(rng, BLUE_INK if seed % 2 else BLACK_INK)
    for text, em_px, is_signature in [("Maple Court Rentals #4", FILL_IN_EM_PX, False),
                                      ("1,450.00", FILL_IN_EM_PX, False),
                                      ("Dana R. Whitfield", SIGNATURE_EM_PX, True)]:
        canvas = transparent_canvas()
        box = draw_handwritten_field(canvas, text, writer, em_px, (40, 140), 1200, rng, is_signature=is_signature)
        assert box is not None
        assert box == ink_bbox(canvas), (text, writer.font_id, writer.signature_font_id)


def test_box_is_clipped_to_ink_that_lands_on_the_canvas():
    rng = np.random.default_rng(3)
    writer = sample_writer(rng, BLACK_INK, ["kalam"], ["allura"])
    canvas = transparent_canvas(300, 60)
    box = draw_handwritten_field(canvas, "Overflowing payee name", writer, FILL_IN_EM_PX, (-40, 50), 1200, rng)
    assert box == ink_bbox(canvas)
    assert box[0] == 0 and box[2] == 300


def test_repeated_letters_are_never_identical():
    rng = np.random.default_rng(11)
    writer = sample_writer(rng, BLACK_INK, ["patrick_hand"], ["allura"])
    canvas = transparent_canvas(1400, 200)
    draw_handwritten_field(canvas, "o o o o o o", writer, FILL_IN_EM_PX, (40, 140), 1300, rng)
    inked = (np.asarray(canvas)[..., 3] > ALPHA_INK_THRESHOLD).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inked, connectivity=8)
    crops = []
    for index in range(1, count):
        x, y, width, height, _ = stats[index]
        glyph = np.asarray(canvas)[y:y + height, x:x + width, 3].astype(np.float32)
        crops.append(cv2.resize(glyph, (32, 32), interpolation=cv2.INTER_AREA))
    assert len(crops) == 6
    for first in range(len(crops)):
        for second in range(first + 1, len(crops)):
            assert np.abs(crops[first] - crops[second]).mean() > 3.0


def test_same_seed_draws_the_same_ink():
    images = []
    for _ in range(2):
        rng = np.random.default_rng(42)
        writer = sample_writer(rng, BLUE_INK)
        canvas = transparent_canvas()
        draw_handwritten_field(canvas, "Rent October", writer, FILL_IN_EM_PX, (40, 140), 1200, rng)
        images.append(np.asarray(canvas))
    assert np.array_equal(images[0], images[1])


def test_thinning_reduces_a_thick_bar_to_a_one_pixel_line():
    mask = np.zeros((40, 120), bool)
    mask[15:25, 10:110] = True
    skeleton = thin_to_centreline(mask)
    assert skeleton[:, 30:90].sum(axis=0).max() == 1
    assert centreline_endpoints(skeleton).sum() == 2


def test_spur_pruning_keeps_dots_and_strokes_but_drops_short_branches():
    skeleton = np.zeros((60, 100), bool)
    skeleton[30, 10:90] = True          # main stroke
    skeleton[27:30, 50] = True          # 3 px spur
    skeleton[10, 20] = True             # an i-dot
    pruned = prune_centreline_spurs(skeleton, 4)
    assert pruned[10, 20]
    assert pruned[30, 10:90].all()
    assert not pruned[27:29, 50].any()  # the base pixel touching the stroke may stay; it sits inside the pen radius


def test_one_check_of_handwriting_renders_fast():
    rng = np.random.default_rng(7)
    writer = sample_writer(rng, BLUE_INK)
    canvas = Image.new("RGBA", (1800, 825), (250, 250, 245, 255))
    fields = ["09/25/2026", "Maple Court Property Management", "1,450.00",
              "One thousand four hundred fifty and 00/100", "October rent unit 4B"]
    draw_handwritten_field(canvas, "warm up caches", writer, FILL_IN_EM_PX, (100, 100), 900, rng)
    start = time.perf_counter()
    for row, text in enumerate(fields):
        draw_handwritten_field(canvas, text, writer, FILL_IN_EM_PX, (100, 150 + row * 110), 1300, rng)
    draw_handwritten_field(canvas, "Dana R. Whitfield", writer, SIGNATURE_EM_PX, (1000, 780), 700, rng, is_signature=True)
    assert time.perf_counter() - start < 0.3
