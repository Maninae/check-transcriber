# Check Transcriber — repo map

Three top-level areas, each a separate concern:

- **`app/`** — the static site (HTML/CSS/ES-module JS), deployed to GitHub Pages via `.github/workflows/deploy-pages.yml`. Read `app/CLAUDE.md` before touching anything in here — it covers the module map, the Content-Security-Policy invariants, the service worker's caching contract, and one non-obvious browser bug workaround in `app/js/engine_loader.js` that is load-bearing (its own docstring explains it; don't "clean it up" back to `async`/`await` without re-reading that first).
- **`synth/`** — Python synthetic-check-data generation (the mock-check regression set described in the product spec's section 8). Owned by separate work; not present as of this commit.
- **`experiments/`** — ML inference and evaluation for later processing-pipeline milestones. Not started.

Dependencies flow one direction: `synth/` and `experiments/` produce data/models that a later milestone of `app/` consumes; `app/` never depends on Python at runtime (it's static files with zero build step, by design — see the product spec's "zero maintenance burden" constraint).

## Where things live

- Product spec (goals, constraints, full UX flow, processing pipeline, all milestones): `docs/SPEC.md`.
- App architecture, CSP/SW invariants, "how to add the next milestone" hook: `app/CLAUDE.md`.
- Repo-level pointer for humans: `README.md`.

## What's built so far

Milestone 1 (the skeleton): drop zone with all three input doors (paste/drag/click), EXIF orientation handling, HEIC detection, the trust sentence, OpenCV.js/Tesseract.js loading with a readiness line, and service-worker offline caching. See `app/CLAUDE.md` for the module-by-module breakdown.
