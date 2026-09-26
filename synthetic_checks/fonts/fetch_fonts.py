"""Download every registered font (plus its license file) into `paths.FONT_DIR`.

Usage: `python -m synthetic_checks.fonts.fetch_fonts`. Idempotent: files already present are skipped.

- google/fonts keeps the license next to the font: `OFL.txt` in `ofl/`, `LICENSE.txt` in `apache/`.
- A few google/fonts folders (Tinos) carry no license text; their `METADATA.pb`, which records
  the license, is saved instead and a warning is logged.
- GnuMICR ships `COPYING` (GPL-2); it is fetched but never bundled anywhere.
"""

import logging
import urllib.error
import urllib.request
from pathlib import Path

from synthetic_checks.fonts.font_registry import APACHE_LICENSE, FONT_SPECS, OFL_LICENSE, FontSpec
from synthetic_data_paths import FONT_DIR

logger = logging.getLogger(__name__)

LICENSE_FILENAME_BY_LICENSE = {
    OFL_LICENSE: "OFL.txt",
    APACHE_LICENSE: "LICENSE.txt",
}
GPL_LICENSE_FILENAME = "COPYING"
GOOGLE_FONTS_METADATA_FILENAME = "METADATA.pb"
HTTP_NOT_FOUND = 404


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
        fetch_license_text(spec, font_path.parent)


def fetch_license_text(spec: FontSpec, local_directory: Path) -> None:
    """Save the upstream license file next to the font, or METADATA.pb when upstream has none."""
    upstream_directory_url = spec.source_url.rsplit("/", 1)[0]
    license_path = local_directory / license_filename_for(spec)
    metadata_path = local_directory / GOOGLE_FONTS_METADATA_FILENAME
    if license_path.exists() or metadata_path.exists():
        return
    try:
        download_file(f"{upstream_directory_url}/{license_path.name}", license_path)
    except urllib.error.HTTPError as http_error:
        if http_error.code != HTTP_NOT_FOUND or spec.license_name not in LICENSE_FILENAME_BY_LICENSE:
            raise
        logger.warning("%s has no %s upstream; saving %s (records the license) instead",
                       spec.font_id, license_path.name, GOOGLE_FONTS_METADATA_FILENAME)
        download_file(f"{upstream_directory_url}/{GOOGLE_FONTS_METADATA_FILENAME}", metadata_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch_all_fonts()
