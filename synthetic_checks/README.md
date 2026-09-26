# synthetic_checks

Renders one flat fake US check at 300 dpi: 66 templates over 6 layout families, paper and security print, laser fill-ins, and ballpoint handwriting drawn by a pen model. The label holds every field's text and tight box, and says which fields are handwritten. Every name, bank, amount and routing number is invented; routing numbers deliberately fail the ABA checksum. Module map and invariants: [CLAUDE.md](CLAUDE.md).

```
# fonts (once; downloaded to the data drive, never committed)
python -m synthetic_checks.fonts.fetch_fonts

# print-ready Letter PDF (clean stock) + one PNG per page + labels CSV/JSON keyed by printed serial
python -m synthetic_checks.print_sheets --output DIR --pages 4
# ... only this seed's eval-split templates, fonts, payees and banks (the real-photo gold set)
python -m synthetic_checks.print_sheets --output DIR --pages 12 --print-pool eval --seed 1
# ... hand-fill: handwritten fields left blank, a strip under each check says what to write, 3 writers (serials -A/-B/-C)
python -m synthetic_checks.print_sheets --output DIR --pages 6 --print-pool eval --seed 1 --hand-fill --writers 3
```

Print `print_sheets.pdf` at 100% scale, cut along the guides, lay the checks on a bedsheet and photograph them; `print_labels.csv` maps each printed serial (S-0007) to every field's text. Hand-fill sheets are written on by people first (instruction page per writer), and the labels add `writer`, `hand_fill` and `prompt_strip_text`.

## Fonts

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

- Printed fonts (41 ids, some sharing a file at another weight) are split into role pools per check kind and a ~24% hold-out set in `fonts/printed_font_pools.py`; `tests/test_printed_fonts.py` checks each draws every character we print. Tinos has no license text next to it upstream, so `fetch_fonts` saves its `METADATA.pb` (which records OFL) instead. No OFL OCR-A/OCR-B face exists in google/fonts.
- Every handwriting and signature font comes from the google/fonts repo (`ofl/` or `apache/`; URL and license per font in `fonts/font_registry.py`), and `tests/test_handwriting.py` checks each draws every character we write. Six handwriting and two signature candidates were rejected by eye because their letters clog at a real ballpoint width (listed in `font_registry.py`).
- GnuMICR's TTF is a third-party conversion of the hand-coded Type 1 font; its glyphs are E-13B-like but not certified. The MICR line is visual texture for detection and orientation, not a readable bank code line.
