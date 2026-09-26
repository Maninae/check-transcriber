# dataset_builder: fixed datasets from the scene composer

Plans a split build, generates scenes in parallel through `scene_composer.on_demand`, and writes files: photos, annotations, COCO / YOLO exports, the per-field OCR manifest, the manifest, and QA / showcase images. Top of the dependency chain: imports the other three packages, nothing imports it.

## Module map

- Top level:
  - `build_dataset.py` CLI coordinator (plan -> scene stage -> OCR stage -> exports -> manifest; `--print-sheets` delegates to `synthetic_checks.print_sheets`).
  - `build_plan.py` pools + counts, `--plan-only`, the `build_plan.json` resume guard, procedural backgrounds on request.
  - `scene_worker.py` one scene: `compose_scene_on_demand` + photo, YOLO labels, annotation (written last, by rename). `scene_task_runner.py` resume skip + per-scene failure capture. `build_progress.py` ETA + `failures.jsonl`.
  - `manifest.py` the `manifest.json` record (generator version, git commit, command, pools, timings).
  - `showcase_images.py` the five presentation images (contact sheet, full scene, 3x crop vs a real photo, flat check, print page).
- `exports/`: `coco_export.py` (checks with outline + corner keypoints, one category per field), `annotation_exports.py` (YOLO seg / OBB lines, data yamls).
- `ocr/`: `ocr_rectify.py` corners -> 1600 px upright crop + point mapping; `ocr_manifest.py` field crops + JSONL rows with `usable` / `status`.
- `qa/`: `contact_sheet.py` scenes with label polygons drawn, `ocr_crop_grid.py` field crops over their ground-truth text.
- `tests/`: `python -m pytest dataset_builder` from the repo root; `conftest.py` builds one 14-scene dataset per session, shared by every test here.

## Invariants

- The builder never composes a scene itself: every scene is `compose_scene_on_demand(seed, split, index, pools=...)`, so builds and on-demand streams cannot drift (`test_stream_reproduces_the_dataset_builders_scenes`).
- Same arguments, same bytes: a rebuild reproduces every photo, label and export exactly; only `manifest.json`'s run metadata (command, time, commit, timings) changes.
- A scene's annotation JSON exists only when the scene is complete (written last, by rename); `--resume` relies on it, and refuses a plan that differs from `build_plan.json`.
- Pixel convention: scene labels put pixel i's centre at i; COCO and YOLO put it at i + 0.5, so exports add half a pixel (`exports/coco_export.to_pixel_edge_coordinates`). `ocr/ocr_rectify.py` stays in centre coordinates internally and reports `box_in_check_crop` in edge coordinates. Never add the shift anywhere else.
- Labels are amodal: a covered check keeps its full outline in segmentation, and its in-frame COCO corner keypoints are v=2 even when hidden (v means in frame).
- YOLO-OBB: fully in-frame checks write their true corners; partly out-of-frame ones write the rectangle, aligned with the check's top edge, enclosing the visible part (Ultralytics' own `preserve_obb` clip rule). It may overhang the frame within Ultralytics' loader tolerance; never clamp corners one by one.
- OCR field boxes are the photo field quad mapped through the same homography that rectifies the pixels, never re-detected.
- Per-scene and per-crop failures are caught, recorded with the scene id and traceback, and the build continues; everything else fails loud.
