# Check Transcriber

A browser-only tool that turns a phone photo of several rent checks into upright, cropped, readable check images with the key fields pulled out — built for a small nonprofit's bookkeeping. Everything runs in the browser: paste or drop a photo, nothing is ever uploaded anywhere.

Live site (once deployed): `https://maninae.github.io/check-transcriber/`

## Repo layout

- **`app/`** — the static site itself (HTML/CSS/JS, no build step), deployed to GitHub Pages. See `app/CLAUDE.md`.
- **`synthetic_checks/`** — renders one flat fake check (templates, fonts, ballpoint handwriting, security print) plus split rules and true-size print sheets. See [README](synthetic_checks/README.md).
- **`synthetic_backgrounds/`** — the library of surfaces checks are laid on, with the FLUX generator and the CC0 web fetcher. See [README](synthetic_backgrounds/README.md).
- **`scene_composer/`** — composes checks onto a background as a phone photo with exact labels, on demand (one scene from a seed, or a lazy stream). See [README](scene_composer/README.md).
- **`dataset_builder/`** — fixed held-out datasets from the composer: COCO/YOLO exports, a per-field OCR manifest, QA and showcase images. See [README](dataset_builder/README.md).
- **`experiments/`** — ML inference/evaluation work for later processing-pipeline milestones. Not started yet.

Datasets, backgrounds, fonts and model weights live on the data drive (`/Volumes/vega/datasets/check-transcriber/`, `/Volumes/vega/ai-models/`), never in the repo; `synthetic_data_paths.py` has the locations.

## Running the app locally

```
cd app
python3 -m http.server 8000
```
Open `http://localhost:8000/`.

## Synthetic data (Python 3.12+)

Run every command from the repo root:

```
python3.12 -m venv /Volumes/vega/datasets/check-transcriber/venv
/Volumes/vega/datasets/check-transcriber/venv/bin/pip install -r requirements.txt
/Volumes/vega/datasets/check-transcriber/venv/bin/python -m synthetic_checks.fonts.fetch_fonts
/Volumes/vega/datasets/check-transcriber/venv/bin/python -m scene_composer.generate_one --seed 7 --out /tmp/x.jpg --labels /tmp/x.json
/Volumes/vega/datasets/check-transcriber/venv/bin/python -m pytest
```

| Package | License | Why |
|---|---|---|
| Pillow | MIT-CMU (HPND) | text rendering, PDF pages |
| numpy | BSD-3 | all pixel math |
| opencv-python-headless | Apache-2.0 | warps, blur, JPEG, polygon fill |
| Faker | MIT | fake payer names and addresses (seeded per check) |
| pytest | MIT | tests |
| torch, torchvision (CPU) | BSD-3 | optional: runs PCT-Net harmonization |
| einops | MIT | optional: imported by PCT-Net's model code |
| kornia | Apache-2.0 | optional: imported by PCT-Net's color functions |

## Full product spec

The complete spec (goals, constraints, UX flow, processing pipeline, milestones) lives in `docs/SPEC.md`. The synthetic-data plan, contracts and build log live in `docs/SYNTHETIC_DATA_PLAN.md`.

## License

MIT for everything in this repository: the app, the synthetic-data packages, the experiments, and the models we trained ourselves (the check detector, the field localizer, the readers, the orientation classifier).

Two third-party notes:

- The YOLO experiments under `experiments/detection/` were trained with Ultralytics (AGPL-3.0). Those weights are a research reference only and are not shipped in the app.
- The optional handwriting reader is Microsoft's `trocr-small-handwritten`. Its code is MIT, its model card states no licence for the weights, and the weights were fine-tuned on the IAM dataset, whose terms are non-commercial research. The app offers it as a separate opt-in download for that reason; it is not part of the default bundle.
