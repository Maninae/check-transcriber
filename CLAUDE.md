# check-transcriber-synth

Synthetic data for Check Transcriber: fake US checks rendered, composited into phone-photo scenes, and split into datasets. Python 3.12+. Will move into the app repo as `synth/`; these two root docs move with it.

## Module map

- `synth/paths.py`: data-drive locations (env-overridable). Nothing large lives in the repo.
- `synth/render/`: stage 1, one flat check.
  - `check_layout.py` sizes, DPI, normalized field anchors. `check_templates.py` 24 deterministic template designs (design i depends only on i).
  - `check_fields.py` content + label dataclasses. `fake_data.py` invented values (invalid-checksum routing numbers).
  - `render_check.py` blank stock (cached per template) + fill-ins. `text_drawing.py` printed and handwriting-style text with tight ink boxes.
  - `security_pattern.py` paper textures. `fonts.py` registry with licenses. `fetch_fonts.py` downloader.
- `synth/compose/`: stage 2, one scene.
  - `compose_scene.py` coordinator. `scene_config.py` knobs. `placement.py` physical layout in inches (grid, loose overlap, fan; >= 70% visible). `scene_framing.py` fits photo orientation, camera and sheet scale to the laid-out group (fill 70-95%, 10% wide), pushes one check out of frame.
  - `paper_deformation.py` paper height field (curl, corner lift, half/thirds fold, waves). `check_plane_map.py` exact check -> plane map (placement + arc compression + parallax) and its fixed-point inverse. `check_paste.py` inverse warp with RGBA coverage, Lambert shading, height-aware shadow.
  - `perspective.py` camera homography, lens distortion, point transforms, polygon clipping. `scene_check_labels.py` corners, outline, enclosing field quads.
  - `lighting.py` classical photometric effects. `camera_effects.py` the photo-side effect pass. `harmonize.py` optional PCT-Net pass. `scene_label.py` label dataclasses.
- `synth/backgrounds/`: `loader.py` (recursive JPEG/PNG, id = relative path), `procedural.py` (fabric fallback for tests). A FLUX generation script will be dropped in here.
- `synth/dataset/`: stage 3. `build_dataset.py` CLI, `scene_worker.py` per-scene job, `splits.py`, `manifest.py`, `annotation_exports.py` (COCO, YOLO), `contact_sheet.py` visual QA.
- `synth/print/print_sheets.py`: Letter PDFs of true-size checks + labels CSV keyed by printed serial.
- `synth/tests/`: pytest; run `python -m pytest -q` from the root (`pytest.ini` sets the path).

## Invariants

- Every geometric image op has a matching point transform; labels are computed by transforming points, never by re-detecting pixels. Change a warp, change its point transform in the same edit (`perspective.py`, `check_plane_map.py`), and keep `test_corner_polygon_covers_pasted_pixels`, `test_outline_covers_and_hugs_deformed_paper`, `test_field_quad_covers_its_ink_after_curl_and_fold` and `test_inverse_map_is_accurate_under_heavy_deformation` green.
- Paper is warped by the inverse of `CheckPlaneMap.check_to_plane`; labels use the forward map. Never warp paper any other way.
- Scene labels (contract C3): `corners` = 4 physical corners (OBB, keypoints); `outline` = deformed paper edge from TL clockwise (YOLO-seg, COCO); `deformation` = parameters. Render images may be RGBA, alpha = paper coverage (C2).
- Corner order is the check's own TL, TR, BR, BL, so the polygon encodes orientation.
- Splits partition template ids and background ids, never scenes. Scene randomness is `default_rng([seed, split_index, scene_index])`.
- Field boxes in check space are measured from drawn ink (handwriting boxes come from the alpha layer after rotation).
- All data is fake. Routing numbers must fail the ABA checksum (`test_routing_numbers_always_fail_aba_checksum`).
- Absolute imports from `synth.`; no leading underscores; imports at top except the documented optional torch/PCT-Net paths.

## Adding things

- New template variation: extend `TemplateDesign` + `build_template_design`; bump `GENERATOR_VERSION` in `synth/__init__.py` (template ids keep their numbers but their look changes).
- New field: add to `FieldName`, draw it in `render_check`, it flows to scenes and exports automatically.
- New camera effect: a pure function in `lighting.py`, called from `apply_camera_effects_in_place`, recorded in the `effects` dict.
