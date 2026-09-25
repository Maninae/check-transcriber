"""Default filesystem locations; every one is overridable by an environment variable.

Large artifacts (fonts, backgrounds, generated datasets) live on the external drive, never
inside the repo, so the package stays small and public-safe.
"""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CHECK_SYNTH_DATA_ROOT", "/Volumes/vega/datasets/check-transcriber"))
FONT_DIR = Path(os.environ.get("CHECK_SYNTH_FONT_DIR", DATA_ROOT / "fonts"))
BACKGROUND_DIR = Path(os.environ.get("CHECK_SYNTH_BACKGROUND_DIR", DATA_ROOT / "backgrounds"))
PROCEDURAL_BACKGROUND_DIR = BACKGROUND_DIR / "procedural"
SYNTH_OUTPUT_DIR = Path(os.environ.get("CHECK_SYNTH_OUTPUT_DIR", DATA_ROOT / "synth"))
