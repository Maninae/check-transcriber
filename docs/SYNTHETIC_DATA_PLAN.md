# Synthetic check images: plan (owner's working doc)

Goal: scenes a person would mistake for phone photos of rent checks on household surfaces, at a glance and under a 3x crop, with labels exact after every transform. Definition of done is in the owner brief; the short form: 60+ templates across layout families, ballpoint-looking handwriting, paper that looks like paper, room lighting, frame-filling layouts, splits by template + background + handwriting font, COCO / YOLO seg + OBB / per-field OCR manifest exports, print sheets, a ~5,000-scene `v1` build.

## How we judge

- By looking: every unit renders its output and the owner Reads it at full size and at a 3x crop, side by side with a real sample (`samples/sarahhdd-cheque-dz` for phone photos of paper and ballpoint, `samples/ssbi-v1.0.0` for real handwriting crops). Scores do not decide.
- Labels by tests: every geometric op has a point transform; coverage tests prove the check polygon covers the pasted pixels and field boxes cover the ink.

## Contracts (fixed before fan-out; change only with the owner)

- C1 Writer (`synthetic_checks/handwriting/handwriting_writer.py`): `sample_writer(rng, ink_rgb, handwriting_font_ids, signature_font_ids) -> Writer`; `draw_handwritten_field(canvas, text, writer, em_px, baseline_left, max_width_px, rng, is_signature=False) -> box | None`. The handwriting engine may add fields to `Writer`; the renderer only calls these two.
- C2 Render output: `render_check(template, content, rng, dpi, handwriting_font_ids, signature_font_ids) -> (PIL image, CheckLabel)`. The image becomes RGBA: alpha is paper coverage (1 on paper, 0 outside a torn perforation nub or worn corner). Compose must use the alpha when the image is RGBA and treat RGB as fully opaque. `CheckLabel` keeps every current field; `canonical` gains `layout_family`.
- C3 Scene label: `SceneCheckLabel.corners` stays the 4 physical corners (check's own TL, TR, BR, BL). New `outline` (N >= 4 points, photo pixels, the paper's edge after curl or fold, starting at TL, in the check's own clockwise order) is what YOLO-seg and COCO polygons export. New `deformation` dict records what was applied.
- C4 Split pools: a split owns template ids, background ids, handwriting and signature font ids, payee names and bank names. `render_check` receives the split's font pools; `sample_check_content` its payee and bank pools.

## Units

File paths in this table and in the log are the pre-split `synth/` layout they were written in; the module map now lives in each package's CLAUDE.md. `synth/v1/`-style paths are folders on the data drive, which did not move.

| Unit | Owner | Files | Verified means |
|---|---|---|---|
| U0 hygiene | owner | seed fix, FLUX scripts adopted, C1 stub | done, see log |
| U1 handwriting | Opus builder, branch `unit/handwriting` | `render/handwriting*.py`, `render/text_drawing.py`, `render/fonts.py`, `render/fetch_fonts.py` | 3x crop of a filled check beside an SSBI crop reads as ballpoint; repeated letters differ; >= 24 handwriting + >= 8 signature fonts, all with the glyphs we write; box tests green |
| U2 stock and print | Opus builder, branch `unit/templates` | `render/check_layout.py`, `check_templates.py`, `render_check.py`, `security_pattern.py`, new paper / print modules, `fake_data.py` | >= 64 templates over >= 5 layout families on one contact sheet; 3x crop of print beside a dz crop shows toner, fibre, misregistration; C2 alpha edge; print sheets render CLEAN stock (no simulated toner or fibre: the real printer and paper add their own) |
| U3 scene geometry | Opus builder, branch `unit/geometry` | `compose/placement.py`, `perspective.py`, new deformation module, `compose_scene.py`, `scene_label.py` | checks fill the frame the way a person frames them; curl and fold look physical; outline + corners + field quads exact (tests) |
| U4 light and harmonization | Opus builder, after U3 | `compose/lighting.py`, new shadow module, `harmonize.py` | hard phone/hand shadows with penumbra, mixed colour temperature, curl shading from geometry, paper never darkened toward the sheet |
| U5 dataset and exports | Opus builder, after U1-U3 | `dataset/*`, `print/*` | splits by template + background + font (test); OCR manifest crops Read and match text; COCO/YOLO load; print sheet page rendered to PNG |
| U6 integration and QA | owner | everything | small build Read at full size and 3x; fresh Opus reviewer on label exactness; fixes |
| U7 v1 build and presentation | owner | data drive | `synth/v1/` ~5,000 scenes + manifest; 5 presentation images |

Backgrounds run all day in the background: FLUX schnell 4-bit, one process, ~100 s per 1024x768 image, 300 images. Build with what exists, rebuild when the batch finishes.

## Log (one line per landed unit)

- U0 (Sep 25): crc32 template seed (per-process `hash()` made stock nondeterministic); FLUX scripts moved to `synth/backgrounds/`, prompts rewritten after "lamp light" painted a lamp; C1 Writer stub wired into `render_check`.
- Carry (Sep 25): U2 adds a temporary `image.convert("RGB")` shim in `dataset/scene_worker.py` ("remove when compose reads alpha (C2)"). At U3 merge: compose reads the RGBA alpha, then delete the shim.
- U3 geometry merged (Sep 25): frame-filling gapped-grid / loose-overlap / fan layouts, curl, folds, 48-point outline labels; 90 tests. Leftover: lonely check in partial last row; field quads enclose (not hug) folded fields.
- U1 handwriting merged (Sep 25): centreline + ballpoint re-ink, per-glyph warp, 34 handwriting / 13 signature OFL+Apache fonts, ~0.1 s per check. Leftover: square stroke ends, legible-name signatures only, alpha-over not multiply.
- U2 stock merged (Sep 25): 66 templates / 6 families (personal classic, date box, name-only scenic, business voucher, three-up, money order), multiply-composited paper/offset/toner/pen layers, perforated edges (RGBA alpha, C2 shim retired), clean print stock by default, one fill mode per check. Leftover: paper mottle slightly blotchy; only 8 printed fonts.
- U5 dataset merged (Sep 25): template split stratified 7/2/2 per family, background + handwriting + signature font pools per split, rectified per-field OCR manifest, COCO outline + field categories, YOLO seg/OBB, resumable builds, clean print sheets with serials + PNG pages.
- U2b typefaces merged (Sep 25): 41 OFL printed fonts in per-role pools, 9 reserved for templates with index % 5 == 4; the template split now deals those to eval/val first so reserved fonts never reach train (test). Fixed the doubled '$ $' on personal stock (fill omits '$' where the stock prints one).
- U4 lighting merged (Sep 25): one key/fill/ambient light for sheet, paper shading and shadows; hand/phone/forearm cast shadows (p=0.35); camera pipeline (~78% sharp); harmonization caps paper brightness; motion blur odd kernels (R1 #1). Leftover: phone shadow reads as a soft blob; fold cues weak under overhead light.
- U6 integration QA (Sep 25): 40-scene build on FLUX backgrounds read at full size and 3x beside dz: reads as phone photos; in-scene 3x crop holds up (fold crease, ballpoint pressure). Tuned wide shots (8%, fill 0.55-0.7) and phone-shadow share (0.3). 0.83 scenes/s on 3 workers.
- Review fixes (Sep 25): background loader scans only `flux/` + `photos/`; splits placed per id by hash (open pools stable under growth from 50 ids, registries rank-dealt); 30 payees + 30 banks, split like fonts; YOLO-OBB fits the visible part as a rectangle in the check's orientation; exports shift labels half a pixel to pixel-edge coordinates; OCR `too_small` status; relocatable data yamls, OBB images linked per file; every split gets a scene from 3 scenes; v0.4.0.
- Backgrounds (Sep 25, Owen): FLUX stopped at 13 accepted images (~100-400 s each was too slow); switching to ~300 CC0/public-domain web images (Poly Haven, ambientCG, Openverse cc0/pdm, Wikimedia Commons) under backgrounds/web/ with SOURCES.jsonl; U8 builder fetching, owner screens.
- Background mix (Sep 25): background_traits.py tags each background lit-photo vs flat swatch and soft vs hard (web SOURCES.jsonl, FLUX slugs); scenes pick lit photos 60% of the time when the split has any, soft surfaces x2; surface_relief.py gives flat soft swatches soft folds + wrinkles (Lambert, tanh-squashed gain). Pool: 13 FLUX + 271 CC0 web.
- Pre-v1 fixes (Sep 25): paper tint from coloured surfaces capped at +-6% per channel (a purple rug dyed the checks); paper formation 1.6%->0.6% and tint drift 1.2%->0.5% (stock read as parchment); security-pattern patchiness halved. v0.5.0.
- U7 v1 built (Sep 25): /Volumes/vega/datasets/check-transcriber/synth/v1, 5,000 scenes / 25,303 checks (train 3,500 / val 750 / eval 750), 0 failed, 6,353 s on 8 workers, 14 GB; OCR rows val 40,740 (26,422 usable) and eval 40,166 (25,473 usable). Gold-eval print run: synth/print/gold-eval-v1 (12 pages, 33 checks, eval pools, seed 1). Showcase: synth/v1/showcase/.
- Package split (Sep 25, Owen): `synth/` became four top-level packages (`synthetic_checks/`, `synthetic_backgrounds/`, `scene_composer/`, `dataset_builder/`) plus `synthetic_data_paths.py`; `scene_composer.on_demand` is now the one composition path (`compose_scene_on_demand`, `SyntheticSceneStream`, `generate_one` CLI) and the dataset worker only writes its output. A 6-scene build is byte-identical before and after.
- Close-up framing (Sep 26, Owen: real photos are closer than v1): `framing_regime` wide / close / single (scene_composer/geometry/framing_regimes.py). close = 2-6 checks (mode 5-6) at 85-98% fill, tight or touching gaps, slight overlaps, a check cut by the frame in ~45% of scenes; single = one check at 70-100% fill, any rotation, cropped-to-check / corner-cut / steep-angle variants. Wide is byte-identical to v0.5.0 (golden-hash test; a 9-scene build's 247 files match). Per-scene mix in builds and streams (default wide 40 / close 45 / single 15), `--pools-from`, `--only-split`; v0.6.0. Built synth/v1.1-closeup-eval: 600 eval scenes on v1's eval pools only (close 450 / single 150, seed 11), 2,270 checks, 0 failed, 1,131 s on 2 workers; OCR too_small 0.14% of rows (close) and 0% (single) vs 25.5% in v1 eval; close checks >= 900 px wide 95.3%.
