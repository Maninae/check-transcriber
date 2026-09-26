# Check Transcriber — repo map

Six top-level areas, each a separate concern:

- **`app/`** — the static site (HTML/CSS/ES-module JS), deployed to GitHub Pages via `.github/workflows/deploy-pages.yml`. Read `app/CLAUDE.md` before touching anything in here — it covers the module map, the Content-Security-Policy invariants, the service worker's caching contract, and one non-obvious browser bug workaround in `app/js/engine_loader.js` that is load-bearing (its own docstring explains it; don't "clean it up" back to `async`/`await` without re-reading that first).
- **`synthetic_checks/`** — one flat fake check: templates, layout families, fonts, ballpoint handwriting, fake data, security print; also the split rules (`splits.py`) and print sheets. See `synthetic_checks/CLAUDE.md`.
- **`synthetic_backgrounds/`** — the background library: loader, surface traits, FLUX generator, CC0 web fetcher. See `synthetic_backgrounds/CLAUDE.md`.
- **`scene_composer/`** — checks + background -> phone-photo scene with exact labels; the one composition path is `on_demand.compose_scene_on_demand` (plus `SyntheticSceneStream`, `generate_one`). See `scene_composer/CLAUDE.md`.
- **`dataset_builder/`** — fixed split datasets, COCO/YOLO exports, OCR manifest, QA; one consumer of the composer. See `dataset_builder/CLAUDE.md`.
- **`experiments/`** — ML inference and evaluation for later processing-pipeline milestones. Not started.

Dependencies flow one direction: `dataset_builder` -> `scene_composer` -> (`synthetic_checks`, `synthetic_backgrounds`) -> `synthetic_data_paths.py` (data-drive locations, env-overridable). Nothing lower imports anything higher, including in tests. The Python packages and `experiments/` produce data/models that a later milestone of `app/` consumes; `app/` never depends on Python at runtime (it's static files with zero build step, by design — see the product spec's "zero maintenance burden" constraint).

## Where things live

- Product spec (goals, constraints, full UX flow, processing pipeline, all milestones): `docs/SPEC.md`.
- Synthetic-data plan, contracts C1-C4 and build log: `docs/SYNTHETIC_DATA_PLAN.md`.
- App architecture, CSP/SW invariants, "how to add the next milestone" hook: `app/CLAUDE.md`.
- Python: one `requirements.txt` and `pytest.ini` at the root; run commands and `python -m pytest` from the root with the venv at `/Volumes/vega/datasets/check-transcriber/venv/`. Absolute imports from the package names; data (datasets, backgrounds, fonts, weights) stays on `/Volumes/vega`.
- Repo-level pointer for humans: `README.md`.

## What's built so far

Milestone 1 (the skeleton): drop zone with all three input doors (paste/drag/click), EXIF orientation handling, HEIC detection, the trust sentence, OpenCV.js/Tesseract.js loading with a readiness line, and service-worker offline caching. See `app/CLAUDE.md` for the module-by-module breakdown. Synthetic data v1 (5,000 held-out scenes, exports, print sheets) is built; see `docs/SYNTHETIC_DATA_PLAN.md`.
