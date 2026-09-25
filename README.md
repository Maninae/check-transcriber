# Check Transcriber synthetic data generator

Generates labeled "phone photo of several checks on a bedsheet" scenes for Check Transcriber, because no public dataset has that input (see the dataset research report). Every name, address, bank, amount and routing number is invented; routing numbers deliberately fail the ABA checksum.

## Three stages

| Stage | Entry | Output |
|---|---|---|
| 1. Render one check | `synth.render.render_check.render_check` | flat 300 dpi RGBA check (alpha = paper, torn perforation) on 66 templates over 6 layout families + label (every field's text, tight box, handwritten flag, template id, layout family) |
| 2. Compose a scene | `synth.compose.compose_scene.compose_scene` | phone-photo JPEG + label (corners, orientation, field quads, visibility) |
| 3. Build a dataset | `python -m synth.dataset.build_dataset` | train/val/eval split by template and background, COCO + YOLO exports, manifest |

## Setup

```
python3.12 -m venv /Volumes/vega/datasets/check-transcriber/venv
/Volumes/vega/datasets/check-transcriber/venv/bin/pip install -r requirements.txt
/Volumes/vega/datasets/check-transcriber/venv/bin/python -m synth.render.fetch_fonts
```

Optional harmonization (`--harmonize`) also needs PCT-Net cloned to `/Volumes/vega/ai-models/harmonizer/pctnet` (`git clone https://github.com/rakutentech/PCT-Net-Image-Harmonization.git pctnet`); its CNN weights ship inside the repo.

## Usage

```
python -m synth.dataset.build_dataset \
  --output /Volumes/vega/datasets/check-transcriber/synth/run-01 \
  --scenes 10000 \
  --seed 1 \
  --workers 5

# no real backgrounds yet: use procedural fabric instead
python -m synth.dataset.build_dataset --output DIR --scenes 200 --procedural-backgrounds 12

# network harmonization of each pasted check (slower, ~1 s extra per scene)
python -m synth.dataset.build_dataset --output DIR --scenes 200 --harmonize --harmonize-blend 0.5

# print-ready Letter PDF of true-size mock checks + labels CSV keyed by printed serial
python -m synth.dataset.build_dataset --output DIR --print-sheets 4

# visual QA: 12 scenes with label polygons drawn
python -m synth.dataset.contact_sheet DIR --out /tmp/sheet.png
```

Backgrounds are read recursively from `/Volumes/vega/datasets/check-transcriber/backgrounds/` (`flux/` generated, `photos/` real). A background's id is its relative path, and ids are what the split assigns.

## Output layout

```
DIR/manifest.json                        seed, counts, template + background ids per split, config, version
DIR/data.yaml                            Ultralytics dataset config
DIR/{train,val,eval}/images/*.jpg
DIR/{train,val,eval}/annotations/*.json  full scene labels
DIR/{train,val,eval}/labels/*.txt        YOLO segmentation (clipped polygon)
DIR/{train,val,eval}/labels_obb/*.txt    YOLO oriented box (4 corners)
DIR/{train,val,eval}/annotations_coco.json  COCO polygons + 4 corner keypoints (TL, TR, BR, BL)
```

Split rule: each template id and each background id belongs to exactly one split, so eval never shares a template or a background with train.

## Dependencies

| Package | License | Why |
|---|---|---|
| Pillow | MIT-CMU (HPND) | text rendering, PDF pages |
| numpy | BSD-3 | all pixel math |
| opencv-python-headless | Apache-2.0 | warps, blur, JPEG, polygon fill |
| Faker | MIT | fake payer names and addresses (seeded per check) |
| pytest | MIT | tests |
| torch, torchvision (CPU) | BSD-3 | optional: runs PCT-Net |
| einops | MIT | optional: imported by PCT-Net's model code |
| kornia | Apache-2.0 | optional: imported by PCT-Net's color functions |

Not used: `augraphy` (MIT, but last release Dec 2023 and it hard-requires full `opencv-python`, numba, scikit-learn and matplotlib). Its camera/paper effects are implemented directly in `compose/lighting.py`.

Harmonization model: PCT-Net CNN (Guerreiro et al., WACV 2023), MPL-2.0 code and bundled weights, used unmodified from its own clone. Harmonizer (Ke et al., ECCV 2022) was rejected because it is CC BY-NC-SA 4.0 and these outputs train a model that ships in a public tool. PCT-Net's weights were trained on iHarmony4, whose images come from COCO, Flickr, MIT-Adobe FiveK and day2night.

## Fonts (downloaded to the data drive, never committed)

| Font | Role | License |
|---|---|---|
| Serif: Libre Baskerville, EB Garamond, Tinos (Times New Roman metrics), Crimson Text, Libre Caslon Text, Old Standard, PT Serif, Merriweather, Arvo | printed | SIL OFL 1.1 |
| Sans: Source Sans 3, PT Sans, Arimo (Arial metrics), Carlito (Calibri metrics), Open Sans, Libre Franklin, Lato, Istok Web | printed | SIL OFL 1.1 |
| Condensed: Oswald, Roboto Condensed, Archivo Narrow, PT Sans Narrow, Barlow Condensed | printed | SIL OFL 1.1 |
| Mono: Courier Prime, Cousine (Courier New metrics), IBM Plex Mono, Share Tech Mono, Anonymous Pro | printed | SIL OFL 1.1 |
| Display: Cinzel, Playfair Display, Marcellus, Archivo Black | printed | SIL OFL 1.1 |
| Caveat, Kalam, Nothing You Could Do, Reenie Beanie, Shadows Into Light, Indie Flower, Patrick Hand, Gochi Hand, Covered By Your Grace, Nanum Pen Script, Gaegu, Architects Daughter, Handlee, Neucha, Sue Ellen Francisco, Annie Use Your Telescope, Just Me Again Down Here, Mynerve, Edu SA Beginner, Edu NSWACT Foundation, Edu VICWANT Beginner, Edu QLD Beginner, The Girl Next Door, Give You Glory, Grape Nuts, Short Stack, Beth Ellen, La Belle Aurore, Dawning of a New Day, Zeyada, Edu TAS Beginner, Marck Script | handwriting | SIL OFL 1.1 |
| Schoolbell, Coming Soon | handwriting | Apache 2.0 |
| Dancing Script, Mr Dafoe, Allura, Mrs Saint Delafield, Kristi, Sacramento, Ruthie, Qwigley, Meddon, Arizonia, Whisper, Ephesis | signature | SIL OFL 1.1 |
| Yellowtail | signature | Apache 2.0 |
| GnuMICR (E-13B) | MICR line | GPL-2.0; only rendered pixels leave the machine |

Printed fonts (41 ids, some sharing a file at another weight) are split into role pools per check kind and a ~24% hold-out set in `synth/render/printed_font_pools.py`; `synth/tests/test_printed_fonts.py` checks each draws every character we print. Tinos has no license text next to it upstream, so `fetch_fonts` saves its `METADATA.pb` (which records OFL) instead. No OFL OCR-A/OCR-B face exists in google/fonts.

Every handwriting and signature font comes from the google/fonts repo (`ofl/` or `apache/` folder; the exact URL and license are in `synth/render/fonts.py`), and `synth/tests/test_handwriting.py` checks each one draws every character we write. Six handwriting and two signature candidates were rejected by eye because their letters clog at a real ballpoint width (listed in `fonts.py`).

GnuMICR's TTF is a third-party conversion of the hand-coded Type 1 font; its glyph shapes are E-13B-like but not certified. The MICR line is visual texture for detection and orientation, not a readable bank code line.
