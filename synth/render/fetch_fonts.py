"""Download every registered font (plus its license file) into `paths.FONT_DIR`.

Usage: `python -m synth.render.fetch_fonts`. Idempotent: files already present are skipped.

- google/fonts keeps the license next to the font: `OFL.txt` in `ofl/`, `LICENSE.txt` in `apache/`.
- GnuMICR ships `COPYING` (GPL-2); it is fetched but never bundled anywhere.
"""

import logging
import urllib.request
from pathlib import Path

from synth.paths import FONT_DIR
from synth.render.fonts import APACHE_LICENSE, FONT_SPECS, OFL_LICENSE, FontSpec

logger = logging.getLogger(__name__)

LICENSE_FILENAME_BY_LICENSE = {
    OFL_LICENSE: "OFL.txt",
    APACHE_LICENSE: "LICENSE.txt",
}
GPL_LICENSE_FILENAME = "COPYING"


def download_file(url: str, destination_path: Path) -> None:
    """Fetch one URL to a path, creating parent directories."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        destination_path.write_bytes(response.read())


def license_filename_for(spec: FontSpec) -> str:
    """Name of the license file that sits next to this font upstream."""
    return LICENSE_FILENAME_BY_LICENSE.get(spec.license_name, GPL_LICENSE_FILENAME)


def fetch_all_fonts(font_dir: Path = FONT_DIR) -> None:
    """Download each font file and the license text that sits next to it upstream."""
    for spec in FONT_SPECS:
        font_path = font_dir / spec.relative_path
        if not font_path.exists():
            logger.info("downloading %s", spec.font_id)
            download_file(spec.source_url, font_path)
        license_path = font_path.parent / license_filename_for(spec)
        if not license_path.exists():
            download_file(spec.source_url.rsplit("/", 1)[0] + "/" + license_path.name, license_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch_all_fonts()
