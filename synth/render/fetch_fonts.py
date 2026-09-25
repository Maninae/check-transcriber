"""Download every registered font (plus its license file) into `paths.FONT_DIR`.

Usage: `python -m synth.fetch_fonts`. Idempotent: files already present are skipped.
"""

import logging
import urllib.request

from synth.render.fonts import FONT_SPECS
from synth.paths import FONT_DIR

logger = logging.getLogger(__name__)


def download_file(url: str, destination_path) -> None:
    """Fetch one URL to a path, creating parent directories."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        destination_path.write_bytes(response.read())


def fetch_all_fonts() -> None:
    """Download each font file and the license text that sits next to it upstream."""
    for spec in FONT_SPECS:
        font_path = FONT_DIR / spec.relative_path
        if not font_path.exists():
            logger.info("downloading %s", spec.font_id)
            download_file(spec.source_url, font_path)
        license_name = "COPYING" if spec.font_id == "gnu_micr" else "OFL.txt"
        license_path = font_path.parent / license_name
        if not license_path.exists():
            download_file(spec.source_url.rsplit("/", 1)[0] + "/" + license_name, license_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch_all_fonts()
