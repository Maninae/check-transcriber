"""Data-drive locations shared by all four synthetic-data packages; each is overridable by an environment variable.

Large artifacts (fonts, backgrounds, generated datasets) live on the external drive, never inside
the repo, so the packages stay small and public-safe. The `CHECK_SYNTH_*` variable names and the
`synth/` dataset folder on the drive predate the package split and are kept so existing data and
shells keep working.
"""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CHECK_SYNTH_DATA_ROOT", "/Volumes/vega/datasets/check-transcriber"))
FONT_DIR = Path(os.environ.get("CHECK_SYNTH_FONT_DIR", DATA_ROOT / "fonts"))
BACKGROUND_DIR = Path(os.environ.get("CHECK_SYNTH_BACKGROUND_DIR", DATA_ROOT / "backgrounds"))
# Only these subfolders of a background root feed builds; `rejected/` (and anything else) is ignored.
ACCEPTED_BACKGROUND_SUBDIRECTORIES = ("flux", "photos", "web")
DATASET_OUTPUT_DIR = Path(os.environ.get("CHECK_SYNTH_OUTPUT_DIR", DATA_ROOT / "synth"))
