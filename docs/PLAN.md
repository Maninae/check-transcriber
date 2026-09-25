# Synthetic check images: plan (owner's working doc)

Goal: scenes a person would mistake for phone photos of rent checks on household surfaces, at a glance and under a 3x crop, with labels exact after every transform. Definition of done is in the owner brief; the short form: 60+ templates across layout families, ballpoint-looking handwriting, paper that looks like paper, room lighting, frame-filling layouts, splits by template + background + handwriting font, COCO / YOLO seg + OBB / per-field OCR manifest exports, print sheets, a ~5,000-scene `v1` build.

## How we judge

- By looking: every unit renders its output and the owner Reads it at full size and at a 3x crop, side by side with a real sample (`samples/sarahhdd-cheque-dz` for phone photos of paper and ballpoint, `samples/ssbi-v1.0.0` for real handwriting crops). Scores do not decide.
- Labels by tests: every geometric op has a point transform; coverage tests prove the check polygon covers the pasted pixels and field boxes cover the ink.

## Contracts (fixed before fan-out; change only with the owner)

- C1 Writer (`synth/render/handwriting_writer.py`): `sample_writer(rng, ink_rgb, handwriting_font_ids, signature_font_ids) -> Writer`; `draw_handwritten_field(canvas, text, writer, em_px, baseline_left, max_width_px, rng, is_signature=False) -> box | None`. The handwriting engine may add fields to `Writer`; the renderer only calls these two.
- C2 Render output: `render_check(template, content, rng, dpi, handwriting_font_ids, signature_font_ids) -> (PIL image, CheckLabel)`. The image becomes RGBA: alpha is paper coverage (1 on paper, 0 outside a torn perforation nub or worn corner). Compose must use the alpha when the image is RGBA and treat RGB as fully opaque. `CheckLabel` keeps every current field; `canonical` gains `layout_family`.
- C3 Scene label: `SceneCheckLabel.corners` stays the 4 physical corners (check's own TL, TR, BR, BL). New `outline` (N >= 4 points, photo pixels, the paper's edge after curl or fold, starting at TL, in the check's own clockwise order) is what YOLO-seg and COCO polygons export. New `deformation` dict records what was applied.
- C4 Split pools: a split owns template ids, background ids and handwriting font ids (signature fonts too). `render_check` receives the split's font pools.

## Units

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
