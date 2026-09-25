# check-transcriber-synth

Synthetic data for Check Transcriber: fake US checks rendered, composited into phone-photo scenes, and split into datasets. Python 3.12+. Will move into the app repo as `synth/`; these two root docs move with it.

## Module map

- `synth/paths.py`: data-drive locations (env-overridable). Nothing large lives in the repo.
- `synth/render/`: stage 1, one flat check.
  - `check_layout.py` sizes, DPI, `LayoutFamily` enum, physical type sizes. `check_templates.py` 66 deterministic template designs over 6 families (design i depends only on i; family = i % 6).
  - Stock (cached per template, dpi, texture): `stock_render.py` paper -> scenic wash -> colour plate (misregistered) -> dark plate, each multiplied in. `families/` one module per layout family, each draws its pre-print on a `stock_canvas.StockCanvas` and returns `field_slots.FieldSlots` (where every fill-in goes); `families/family_helpers.py` shared blocks. `paper_texture.py` fibre, formation, specks, security fibres. `scenic_background.py` faded scenes. `noise_fields.py` shared noise.
  - `check_fields.py` content + label dataclasses. `fake_data.py` invented values (invalid-checksum routing numbers).
  - `render_check.py` (C2) stock + laser layer (`laser_printing.py`, toner model in `print_model.py`) + pen layer (C1, multiplied in) + paper alpha (`paper_edges.py`: perforation, worn corners). `simulate_print_texture=False` is the clean path print sheets use. `text_drawing.py` printed text with tight boxes, plus the ink-compositing helper.
  - Handwriting (a ballpoint pen model, not a font): `handwriting_writer.py` is the C1 contract (`Writer`, `sample_writer`, `draw_handwritten_field`). `handwriting_field_renderer.py` runs the pipeline: `handwriting_line_layout.py` places glyph coverage (glyph by glyph for print hands, word by word for cursive so joins survive), `handwriting_line_warp.py` shears for slant and wanders the baseline and elastically warps at glyph scale, `pen_centreline.py` thins to a 1 px centreline (vectorized Zhang-Suen) and prunes spurs, `pen_ballpoint.py` re-inks it with pressure-varying width and density, blobs and skips. `handwriting_quirks.py` retrace, overrun, signature flourish. `handwriting_font_metrics.py` glyph coverage + x-height / ascender calibration. `handwriting_habits.py`, `handwriting_noise.py` shared leaf helpers.
  - `security_pattern.py` warped, irregular security patterns (fans, rosettes, waves, dot screen, hidden VOID pantograph). `fonts.py` registry with licenses. `fetch_fonts.py` downloader.
- `synth/compose/`: stage 2, one scene.
  - `compose_scene.py` coordinator. `scene_config.py` knobs. `placement.py` physical layout in inches (grid, loose overlap, fan; >= 70% visible). `scene_framing.py` fits photo orientation, camera and sheet scale to the laid-out group (fill 70-95%, 10% wide), pushes one check out of frame.
  - `paper_deformation.py` paper height field (curl, corner lift, half/thirds fold, waves). `check_plane_map.py` exact check -> plane map (placement + arc compression + parallax) and its fixed-point inverse. `check_paste.py` inverse warp with RGBA coverage, Lambert shading, height-aware shadow.
  - `perspective.py` camera homography, lens distortion, point transforms, polygon clipping. `scene_check_labels.py` corners, outline, enclosing field quads.
  - `lighting.py` classical photometric effects. `camera_effects.py` the photo-side effect pass. `harmonize.py` optional PCT-Net pass. `scene_label.py` label dataclasses.
- `synth/backgrounds/`: `loader.py` (recursive JPEG/PNG, id = relative path), `procedural.py` (fabric fallback for tests). A FLUX generation script will be dropped in here.
- `synth/dataset/`: stage 3. `build_dataset.py` CLI coordinator (scene stage, OCR stage, exports). `build_plan.py` pools + counts, `--plan-only`, `build_plan.json` resume guard. `splits.py` C4 partitions. `scene_worker.py` one scene (annotation written last, by rename); `scene_task_runner.py` resume skip + per-scene failure capture; `build_progress.py` ETA + `failures.jsonl`. `ocr_rectify.py` corners -> 1600 px upright crop + point mapping; `ocr_manifest.py` field crops + JSONL rows. `coco_export.py`, `annotation_exports.py` (YOLO seg/OBB, data yamls). `manifest.py`. QA: `contact_sheet.py`, `ocr_crop_grid.py`.
- `synth/print/`: `print_sheets.py` Letter PDF + page PNGs of true-size clean-stock checks + labels keyed by printed serial; `print_page_layout.py` slot geometry and cut guides.
- `synth/tests/`: pytest; run `python -m pytest -q` from the root (`pytest.ini` sets the path).

## Invariants

- Every geometric image op has a matching point transform; labels are computed by transforming points, never by re-detecting pixels. Change a warp, change its point transform in the same edit (`perspective.py`, `check_plane_map.py`), and keep `test_corner_polygon_covers_pasted_pixels`, `test_outline_covers_and_hugs_deformed_paper`, `test_field_quad_covers_its_ink_after_curl_and_fold` and `test_inverse_map_is_accurate_under_heavy_deformation` green.
- Paper is warped by the inverse of `CheckPlaneMap.check_to_plane`; labels use the forward map. Never warp paper any other way.
- Scene labels (contract C3): `corners` = 4 physical corners (OBB, keypoints); `outline` = deformed paper edge from TL clockwise (YOLO-seg, COCO); `deformation` = parameters. Render images may be RGBA, alpha = paper coverage (C2).
- Corner order is the check's own TL, TR, BR, BL, so the polygon encodes orientation.
- Splits partition template ids (per layout family), background ids, handwriting font ids and signature font ids, never scenes; `test_scenes_only_use_their_own_split_pools` reads fonts back from annotations. Scene randomness is `default_rng([seed, split_index, scene_index])`.
- OCR field boxes are the photo field quad mapped through the same homography that rectifies the pixels, never re-detected.
- Ink multiplies paper (subtractive), never alpha-over. Pre-printed labels live on the dark plate, so plate misregistration never moves a box; laser boxes are measured from the toner that landed.
- A check is filled by hand or by software, never mixed (money orders excepted: issuer prints date and amount).
- Field boxes in check space are measured from drawn ink (handwriting boxes come from the final alpha actually composited onto the canvas, after every warp and quirk; `test_returned_box_is_exactly_the_laid_down_ink`).
- Every handwriting/signature font must be SIL OFL or Apache 2.0 from google/fonts and draw every character we write (`test_pen_font_covers_every_character_we_write`). Adding a font: one `google_font(...)` line in `fonts.py`, run `python -m synth.render.fetch_fonts`, look at it in the pen engine before keeping it.
- All data is fake. Routing numbers must fail the ABA checksum (`test_routing_numbers_always_fail_aba_checksum`).
- Absolute imports from `synth.`; no leading underscores; imports at top except the documented optional torch/PCT-Net paths.

## Adding things

- New layout family: add to `LayoutFamily` + `FAMILY_SIZE_KIND`, write `families/<name>.py` returning `FieldSlots`, register in `families/__init__.py`; `test_layout_families.py` picks it up.
- New template variation: extend `TemplateDesign` + `build_template_design`; bump `GENERATOR_VERSION` in `synth/__init__.py` (template ids keep their numbers but their look changes).
- New field: add to `FieldName`, draw it in `render_check`, it flows to scenes and exports automatically.
- New camera effect: a pure function in `lighting.py`, called from `apply_camera_effects_in_place`, recorded in the `effects` dict.
