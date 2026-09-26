# synthetic_backgrounds

The surfaces synthetic checks are photographed on: bedsheets, blankets, rugs, tables. Backgrounds are read recursively from the accepted subfolders of `/Volumes/vega/datasets/check-transcriber/backgrounds/`: `flux/` generated, `photos/` own real photos, `web/` CC0 / public-domain downloads. `rejected/` and anything else is ignored. A background's id is its path relative to that root, and ids are what the split assigns. Module map and invariants: [CLAUDE.md](CLAUDE.md).

## Generating (local FLUX.1-schnell 4-bit via mflux)

One process at a time, ~100 s per 1024x768 image on the 16 GB Mac mini, resumable:

```
~/.claude/skills/flux-image-gen/.venv/bin/python -m synthetic_backgrounds.generate_flux_backgrounds \
  --start 0 \
  --count 300 \
  --out /Volumes/vega/datasets/check-transcriber/backgrounds/flux
```

Screen each image by eye; move failures to `backgrounds/rejected/` with a line in `REJECTED.md`.

## Downloading (keyless; Poly Haven, ambientCG, Openverse, Wikimedia Commons)

Resumable, ~1 h for a full pass:

```
CHECK_SYNTH_FETCH_CONTACT="you@example.org" \
/Volumes/vega/datasets/check-transcriber/venv/bin/python -m synthetic_backgrounds.fetch_web_backgrounds \
  --output /Volumes/vega/datasets/check-transcriber/backgrounds/web
```

- License: CC0 or public domain only (these images train a model that ships publicly). Openverse and Commons are checked per image from their own metadata; Poly Haven and ambientCG are CC0 site-wide. No CC BY, no "free with attribution", no unknown. Pexels, Pixabay and Unsplash are not used (keys or license terms).
- `web/SOURCES.jsonl` has one row per kept file: source, page and image URL, author, license and where it was read, title, category, fetch date, and how the texture was scaled. `web/FETCH_SKIPS.jsonl` logs every candidate dropped automatically (license, too small, flat, oversaturated, near-duplicate, download error) so reruns skip it.
- Textures with a known physical size are tiled 2x2 or centre-cropped towards ~600 mm across, so weave and grain match a phone photo of a few checks on a table.
- Near-duplicates (against everything under `backgrounds/`, rejected included) are caught by a two-band perceptual signature (`web_sources/perceptual_duplicate_index.py`).
- Then screen by eye: reject text, logos, watermarks, people, hands, objects on the surface, room or side views, extreme perspective, and non-surfaces (bark, gravel, walls); move them to `backgrounds/rejected/web/` with a line in `rejected/REJECTED.md`.
