# synthetic_checks: one flat fake check

Renders a single US check (stock, pre-print, laser fill-ins, ballpoint handwriting) and its exact label. Also owns the split rules and true-size print sheets. Depends only on `synthetic_data_paths` (fonts); nothing here imports the other three packages. Python 3.12+.

## Module map

- Top level (the contracts):
  - `render_check.py` (C2) stock + laser layer + pen layer (multiplied in) + paper alpha. Returns an RGBA image (alpha = paper coverage) and a `CheckLabel`. `simulate_print_texture=False` is the clean path print sheets use.
  - `check_fields.py` content + label dataclasses (`FieldName`, `CheckContent`, `CheckLabel`). `check_layout.py` sizes, DPI, `LayoutFamily`, physical type sizes. `check_templates.py` 66 deterministic designs over 6 families (design i depends only on i; family = i % 6), `template_family_by_id`. `field_slots.py` where every fill-in goes (what a family returns).
  - `splits.py` contract C4: which templates, backgrounds, fonts, payees and banks belong to train / val / eval (see Invariants).
  - `print_sheets.py` Letter PDF + page PNGs of true-size clean-stock checks + labels keyed by printed serial; `print_page_layout.py` slot geometry and cut guides.
- `families/`: one module per layout family, each draws its pre-print on a `StockCanvas` and returns `FieldSlots`; `family_helpers.py` shared blocks; `__init__.py` the registry (`FAMILY_DRAWERS`, `PERFORATED_FAMILIES`).
- `stock/`: paper and pre-print, cached per template, dpi, texture. `stock_render.py` paper -> scenic wash -> colour plate (misregistered) -> dark plate, each multiplied in. `stock_canvas.py`, `paper_texture.py` (fibre, formation, specks, security fibres), `paper_edges.py` (perforation, worn corners), `scenic_background.py`, `security_pattern.py` (fans, rosettes, waves, dot screen, hidden VOID pantograph), `noise_fields.py`.
- `printed_text/`: `laser_printing.py` the laser fill-in layer, `print_model.py` toner model, `text_drawing.py` printed text with tight boxes plus the ink-compositing helper.
- `handwriting/`: a ballpoint pen model, not a font. `handwriting_writer.py` is the C1 contract (`Writer`, `sample_writer`, `draw_handwritten_field`). `handwriting_field_renderer.py` runs the pipeline: `handwriting_line_layout.py` places glyph coverage (glyph by glyph for print hands, word by word for cursive so joins survive), `handwriting_line_warp.py` shears for slant, wanders the baseline, warps at glyph scale, `pen_centreline.py` thins to a 1 px centreline (vectorized Zhang-Suen) and prunes spurs, `pen_ballpoint.py` re-inks it with pressure-varying width and density, blobs and skips. `handwriting_quirks.py` retrace, overrun, signature flourish. `handwriting_font_metrics.py` glyph coverage + x-height / ascender calibration. `handwriting_habits.py`, `handwriting_noise.py` shared leaves.
- `fonts/`: `font_registry.py` every font with its license (variable fonts pinned to a named instance per id), `load_font`, `font_ids_with_role`. `fetch_fonts.py` downloader (CLI). `printed_font_pools.py` printed-font styles, role pools per check kind (header / body / label / fill) and the hold-out set.
- `content/`: `fake_data.py` invented values (routing numbers fail the ABA checksum), `fake_payees_and_banks.py` 30 payees and 30 banks, `amount_words.py`.
- `tests/`: `python -m pytest synthetic_checks` from the repo root.

## Invariants

- Ink multiplies paper (subtractive), never alpha-over. Pre-printed labels live on the dark plate, so plate misregistration never moves a box; laser boxes are measured from the toner that landed.
- A check is filled by hand or by software, never mixed (money orders excepted: issuer prints date and amount).
- Field boxes in check space are measured from drawn ink (handwriting boxes come from the final alpha composited onto the canvas, after every warp and quirk; `test_returned_box_is_exactly_the_laid_down_ink`).
- Every handwriting/signature font is SIL OFL or Apache 2.0 from google/fonts and draws every character we write (`test_pen_font_covers_every_character_we_write`).
- Printed fonts: every PRINTED registry id has a `PrintedFontProfile` (the catalog raises otherwise) and draws every character we print (`test_printed_font_covers_every_character_we_print`). Held-out printed fonts appear only on templates with index % 5 == 4 (`test_holdout_fonts_only_appear_on_holdout_class_templates`); serial and microprint fonts are fixed on every template (`SHARED_PRINTED_FONT_IDS`).
- Splits (C4) partition template ids (per layout family), background ids, handwriting and signature font ids, payee names and bank names, never scenes. Each id is placed by its own blake2b score: open pools (backgrounds, 50+ ids) by threshold, so growing them never moves an id (`test_adding_backgrounds_never_moves_an_existing_one`); closed registries and small pools by rank, for exact proportions. Templates carrying eval-reserved fonts never reach train.
- All data is fake. Routing numbers must fail the ABA checksum (`test_routing_numbers_always_fail_aba_checksum`).
- Absolute imports (`from synthetic_checks.handwriting.handwriting_writer import ...`); no leading underscores; imports at the top.

## Adding things

- New layout family: add to `LayoutFamily` + `FAMILY_SIZE_KIND`, write `families/<name>.py` returning `FieldSlots`, register it in `families/__init__.py`; `tests/test_layout_families.py` picks it up.
- New template variation: extend `TemplateDesign` + `build_template_design`; bump `GENERATOR_VERSION` in `scene_composer/__init__.py` (template ids keep their numbers but their look changes).
- New field: add to `FieldName`, draw it in `render_check`; it flows to scenes and exports automatically.
- New handwriting font: one `google_font(...)` line in `fonts/font_registry.py`, run `python -m synthetic_checks.fonts.fetch_fonts`, look at it in the pen engine before keeping it.
- New printed font: `google_font(...)` in `fonts/font_registry.py`, a profile line in `fonts/printed_font_pools.py`, fetch, look at it on the contact sheet, bump `GENERATOR_VERSION`.
