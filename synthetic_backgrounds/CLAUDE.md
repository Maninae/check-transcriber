# synthetic_backgrounds: the background library

Surfaces checks are laid on, how a scene picks one, and the two tools that grow the library. Images live on the data drive (`synthetic_data_paths.BACKGROUND_DIR`); this package never imports `synthetic_checks`, `scene_composer` or `dataset_builder`.

## Module map

- `loader.py`: recursive JPEG/PNG under the accepted subfolders `flux/` + `photos/` + `web/` only (`synthetic_data_paths.ACCEPTED_BACKGROUND_SUBDIRECTORIES`); id = path relative to the root; `cover_crop`, `load_background_rgb`.
- `background_traits.py`: tags each background lit-photo vs flat swatch and soft vs hard (from `web/SOURCES.jsonl`, FLUX filename slugs, folder); `choose_background` favours lit photos and soft surfaces, drawing only from the pool it is given.
- `surface_relief.py`: soft folds and wrinkles for flat soft swatches (applied at compose time). `procedural.py`: synthetic fabric for tests and runs without real backgrounds.
- FLUX: `generate_flux_backgrounds.py` (CLI, runs in the flux-image-gen venv) + `background_prompts.py` (prompt vocabulary).
- Web: `fetch_web_backgrounds.py` coordinator CLI (license gate -> download -> scale normalization -> filters -> dedupe -> JPEG + `SOURCES.jsonl` row; resumable via `FETCH_SKIPS.jsonl` and file names in `web/` and `rejected/web/`). `web_sources/` holds one module per source (`web_sources_polyhaven.py`, `web_sources_ambientcg.py` reading the Color map out of the remote zip by HTTP Range, `web_sources_openverse.py`, `web_sources_commons.py`), `web_photo_search_terms.py` shared terms + title exclusions, and the leaves `web_candidate.py` (candidate record, `SurfaceCategory`, license gate), `web_http.py` (polite urllib client, `HttpRangeFile`), `web_image_processing.py` (filters, tiling/cropping, JPEG), `perceptual_duplicate_index.py` (two-band near-duplicate index).
- `tests/`: `python -m pytest synthetic_backgrounds` from the repo root.

## Invariants

- Only `flux/`, `photos/` and `web/` feed scenes; `rejected/` and anything else never does (`test_background_loader_scans_only_accepted_subfolders`). Background ids are paths relative to the root, and ids are what the split assigns, so never rename a kept file.
- Web backgrounds are CC0 or public domain only, verified from the source's metadata and recorded per file in `web/SOURCES.jsonl`; the gate is `web_candidate.ACCEPTED_SOURCE_LICENSE_CODES` (`tests/test_web_backgrounds.py`). Never add a source whose license is per-image unknown or attribution-bound.
- FLUX paints every noun it reads: the prompt vocabulary never names a device, a light source, furniture, a room, "household" or "indoor", and never asks for tilt.
- Every kept background is screened by eye; failures move to `rejected/` with a line in `rejected/REJECTED.md` (their names stay known, so a rerun never fetches them again).
