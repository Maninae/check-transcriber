# scene_composer: checks + background -> one phone-photo scene with exact labels

Composes rendered checks (`synthetic_checks`) onto a background (`synthetic_backgrounds`) as a phone would photograph them, and serves scenes on demand. `dataset_builder` is one consumer; a training loop or demo is another. Imports only `synthetic_checks`, `synthetic_backgrounds` and `synthetic_data_paths`, never `dataset_builder`.

## Module map

- Top level:
  - `on_demand.py` THE composition path. `compose_scene_on_demand(seed, split, scene_index, ...) -> ComposedScene` draws checks, background and effects from `default_rng([seed, split_index, scene_index])`, in memory. `SyntheticSceneStream` yields a split's scenes lazily. `ComposedScene.write_jpeg` is how every scene file is saved.
  - `framing_regime_mix.py` which framing regime each scene gets from a mix (default wide 40 / close 45 / single 15): exact counts per block of 20 scene indices, shuffled by its own rng, so streams and builds agree and wide scenes are untouched.
  - `scene_ingredient_pools.py` `SceneIngredientPools` (a split's templates, backgrounds, fonts, payees, banks): from a planned build (`ingredient_pools_for_split`) or planned on the spot over the whole library (`library_ingredient_pools`, cached per process).
  - `generate_one.py` CLI over one scene. `compose_scene.py` the coordinator (the 9-step pipeline in its docstring). `scene_config.py` knobs. `scene_label.py` label dataclasses (contract C3).
  - `__init__.py` `GENERATOR_VERSION`, the version of the whole generator.
- `geometry/`: `framing_regimes.py` the three regimes (`wide` = v1, `close` = 2-6 checks at 85-98% fill, `single` = one check at 70-100%, any rotation, cropped / corner-cut / steep variants) and each one's `FramingPolicy`. `placement.py` wide layout in inches (grid, loose overlap, fan; >= 70% visible). `closeup_placement.py` close layouts (tight or touching grids, slight overlaps, >= 80% visible) and the single check's rotation. `scene_framing.py` fits photo orientation, camera and sheet scale to the group by the regime's policy, pushes one check out of frame (edge or corner). `paper_deformation.py` paper height field (curl, corner lift, half/thirds fold, waves). `check_plane_map.py` exact check -> plane map (placement + arc compression + parallax) and its fixed-point inverse. `check_paste.py` inverse warp with RGBA coverage. `perspective.py` camera homography, lens distortion, point transforms, polygon clipping. `scene_check_labels.py` corners, outline, enclosing field quads.
- `lighting/`: one light per scene, applied once to the whole sheet-plane canvas in linear light. `scene_light.py` `SceneLight` (key aimed from the background's bright side, black-body colours, optional fill, ambient, white balance). `scene_irradiance.py` `LightBuffers` + `apply_scene_light`. `paper_shading.py` what a check writes into the buffers (Lambert from the height map, crease ridge, drop shadow, contact occlusion). `cast_shadows.py` + `occluder_silhouettes.py` phone / hand / forearm shadows with height-dependent penumbra, never more than half a check in umbra. `baked_background_light.py` transfers the background photo's baked light and colour cast onto the paper albedo (tint capped).
- `camera/`: `camera_pipeline.py` pure ops in phone order (auto-exposure, vignette, sheen, defocus / centred motion blur, sensor noise in linear, local tone map, contrast, chroma + luma denoise, sharpening, JPEG). `camera_effects.py` samples and orders them and records `effects`.
- `harmonization/`: `harmonize.py` optional PCT-Net pass on albedo (torch imported only when enabled); `harmonize_color_transfer.py` caps its chroma shift and clamps its luminance shift so paper never greys toward the sheet.
- `tests/`: `python -m pytest scene_composer` from the repo root.

## Invariants

- One composition path: everything that produces a scene calls `compose_scene_on_demand`; the dataset worker only adds file writes (`test_stream_reproduces_the_dataset_builders_scenes`). Its rng draw order (check count, per check template + content + render, background, relief, compose) is the dataset's determinism; reordering it changes every scene, so bump `GENERATOR_VERSION` if you must.
- Scenes draw only from their split's `SceneIngredientPools` (contract C4; `test_stream_honours_split_pools`). `library_ingredient_pools` plans exactly as `dataset_builder.build_plan.make_build_plan` does for the same seed, background root and template count.
- Every geometric image op has a matching point transform; labels are computed by transforming points, never by re-detecting pixels. Change a warp, change its point transform in the same edit (`perspective.py`, `check_plane_map.py`), and keep `test_corner_polygon_covers_pasted_pixels`, `test_outline_covers_and_hugs_deformed_paper`, `test_field_quad_covers_its_ink_after_curl_and_fold` and `test_inverse_map_is_accurate_under_heavy_deformation` green.
- Paper is warped by the inverse of `CheckPlaneMap.check_to_plane`; labels use the forward map. Never warp paper any other way.
- Wide is frozen: its policy holds the v1 constants and every close/single-only rng draw is guarded, so a wide scene is byte-identical to v0.5.0 (`test_wide_regime_is_byte_identical_to_the_pre_regime_generator`, golden hashes). Labels carry `effects.framing.framing_regime` only off wide; read it with `framing_regime_of_label` (v1 labels have none and are wide).
- Scene labels (C3): `corners` = 4 physical corners in the check's own TL, TR, BR, BL order (so the polygon encodes orientation); `outline` = deformed paper edge from TL clockwise; `deformation` = parameters. Check images may be RGBA, alpha = paper coverage (C2); RGB is treated as opaque.
- Pixel convention: scene labels put pixel i's centre at i (OpenCV's convention, matching every warp). Only `dataset_builder` exports shift to edges.
- Labels are amodal: a covered check keeps its full outline; occlusion lives in `visible_fraction`.

## Adding things

- New camera effect: a pure function in `camera/camera_pipeline.py` that moves no pixel (symmetric kernels only; `tests/test_camera_pipeline.py` guards centroids), sampled and called from `apply_camera_effects`, recorded in the `effects` dict.
- New light behaviour: extend `SceneLight` / `LightBuffers`; anything that darkens or brightens a region writes a factor into the buffers rather than multiplying colours, so sheet, paper and shadows stay in one light. Record parameters in `SceneLight.to_dict`.
- New framing regime: a `FramingRegime` member, a `FRAMING_POLICIES` entry, a check-count branch in `compose_scene.sample_check_count` and a layout branch in `plan_layout_for_regime`; extend `tests/test_framing_regimes.py` (corner probe, outline coverage, fill).
- New on-demand consumer: iterate `SyntheticSceneStream(seed, split=...)` or call `compose_scene_on_demand`; never re-implement the draw sequence.
