# Check Transcriber

A browser-only tool that turns a phone photo of several rent checks into upright, cropped, readable check images with the key fields pulled out — built for a small nonprofit's bookkeeping. Everything runs in the browser: paste or drop a photo, nothing is ever uploaded anywhere.

Live site (once deployed): `https://maninae.github.io/check-transcriber/`

## Repo layout

This repo holds three areas:

- **`app/`** — the static site itself (HTML/CSS/JS, no build step), deployed to GitHub Pages. This is the only area that exists yet; see `app/CLAUDE.md` for its architecture and `app/README.md`-equivalent details.
- **`synth/`** — Python synthetic-check-data generation pipeline: renders fake checks, composites them onto household-surface photos as phone-photo scenes with exact labels, builds held-out train/val/eval splits with COCO/YOLO exports and a per-field OCR manifest, and prints true-size mock checks for a real-photo eval set. See `synth/README.md` and `synth/CLAUDE.md`. Datasets and backgrounds live outside the repo.
- **`experiments/`** — ML inference/evaluation work for later processing-pipeline milestones. Not started yet.

## Running the app locally

```
cd app
python3 -m http.server 8000
```
Open `http://localhost:8000/`.

## Full product spec

The complete spec (goals, constraints, UX flow, processing pipeline, milestones) lives in `docs/SPEC.md`. This repo implements milestone 1 (the skeleton) as of the initial commit.
