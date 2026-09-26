"""Fetch CC0 / public-domain household-surface backgrounds from keyless web sources.

Usage (the project venv; stdlib urllib, no API keys):
    python -m synthetic_backgrounds.fetch_web_backgrounds \
        --output /Volumes/vega/datasets/check-transcriber/backgrounds/web

Pipeline per candidate, in a deterministic order (source order, then each source's own order):
license gate -> download -> decode -> scale normalization -> automatic filters -> perceptual
dedupe against everything already under the background root (accepted folders and rejected/) ->
long edge <= 2048 px -> JPEG q92 as `web__<source>__<slug>__<id>.jpg` -> one row in SOURCES.jsonl.

- Resumable: a candidate whose file exists in `web/` or `rejected/web/`, or that is logged in
  `web/FETCH_SKIPS.jsonl` (license, filter, duplicate, download error), is not fetched again.
- A human still screens every kept image by eye (contact sheets) and moves misfits to
  `rejected/web/` with a line in `rejected/REJECTED.md`.
"""

import argparse
import datetime
import json
import logging
import sys
import urllib.error
import zipfile
from collections.abc import Callable
from pathlib import Path

from synthetic_backgrounds.loader import BACKGROUND_SUFFIXES, load_background_rgb
from synthetic_backgrounds.web_sources.perceptual_duplicate_index import PerceptualDuplicateIndex, perceptual_signature
from synthetic_backgrounds.web_sources.web_candidate import WebBackgroundCandidate, WebBackgroundSource, accepted_license
from synthetic_backgrounds.web_sources.web_http import PoliteHttpClient
from synthetic_backgrounds.web_sources.web_image_processing import (decode_image_rgb, encode_jpeg, limit_long_edge,
                                                                    normalize_physical_scale, rejection_reason)
from synthetic_backgrounds.web_sources.web_sources_ambientcg import download_ambientcg_color_map, list_ambientcg_candidates
from synthetic_backgrounds.web_sources.web_sources_commons import list_commons_candidates
from synthetic_backgrounds.web_sources.web_sources_openverse import list_openverse_candidates
from synthetic_backgrounds.web_sources.web_sources_polyhaven import list_polyhaven_candidates
from synthetic_data_paths import BACKGROUND_DIR

logger = logging.getLogger(__name__)

SOURCES_LOG_NAME = "SOURCES.jsonl"
SKIPS_LOG_NAME = "FETCH_SKIPS.jsonl"
REJECTED_WEB_SUBDIRECTORY = Path("rejected") / "web"

CANDIDATE_LISTERS: dict[WebBackgroundSource, Callable[[PoliteHttpClient], list[WebBackgroundCandidate]]] = {
    WebBackgroundSource.POLYHAVEN: list_polyhaven_candidates,
    WebBackgroundSource.AMBIENTCG: list_ambientcg_candidates,
    WebBackgroundSource.OPENVERSE: list_openverse_candidates,
    WebBackgroundSource.COMMONS: list_commons_candidates,
}


def download_candidate_bytes(client: PoliteHttpClient, candidate: WebBackgroundCandidate) -> bytes:
    """Encoded image bytes: ambientCG's Color map comes out of a zip, every other source is a direct URL."""
    if candidate.source is WebBackgroundSource.AMBIENTCG:
        return download_ambientcg_color_map(client, candidate)
    return client.get_bytes(candidate.image_url)


def build_existing_duplicate_index(background_root: Path) -> PerceptualDuplicateIndex:
    """Signatures of every image already under the background root (accepted and rejected), so nothing returns twice."""
    index = PerceptualDuplicateIndex()
    for image_path in sorted(background_root.rglob("*")):
        if image_path.suffix.lower() in BACKGROUND_SUFFIXES and image_path.is_file() and not image_path.name.startswith("."):
            index.add(perceptual_signature(load_background_rgb(image_path)), image_path.relative_to(background_root).as_posix())
    logger.info("signed %d existing backgrounds under %s", len(index), background_root)
    return index


def read_logged_file_names(log_path: Path) -> set[str]:
    """`file` values of a JSONL log (empty when the log does not exist yet)."""
    if not log_path.exists():
        return set()
    return {json.loads(line)["file"] for line in log_path.read_text().splitlines() if line.strip()}


def append_json_line(log_path: Path, row: dict) -> None:
    """Append one JSON row."""
    with open(log_path, "a") as log_file:
        log_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_candidate(client: PoliteHttpClient, candidate: WebBackgroundCandidate, output_directory: Path,
                      duplicate_index: PerceptualDuplicateIndex, fetch_date: str) -> tuple[dict | None, str | None]:
    """Fetch and keep one candidate. Returns (SOURCES row, None) when kept or (None, skip reason)."""
    normalized_license = accepted_license(candidate)
    if normalized_license is None:
        return None, f"license not CC0/PD: {candidate.source_license_code or 'missing'}"
    image_rgb = decode_image_rgb(download_candidate_bytes(client, candidate))
    processed = normalize_physical_scale(image_rgb, candidate.physical_extent_mm, candidate.is_seamless_texture)
    reason = rejection_reason(processed.image_rgb)
    if reason is not None:
        return None, reason
    final_image = limit_long_edge(processed.image_rgb)
    signature = perceptual_signature(final_image)
    duplicate_of = duplicate_index.find_duplicate(signature)
    if duplicate_of is not None:
        return None, f"near-duplicate of {duplicate_of}"
    output_path = output_directory / candidate.output_file_name
    output_path.write_bytes(encode_jpeg(final_image))
    duplicate_index.add(signature, f"web/{candidate.output_file_name}")
    return {
        "file": candidate.output_file_name, "source": candidate.source.value, "source_id": candidate.source_id,
        "source_page_url": candidate.source_page_url, "image_url": candidate.image_url, "author": candidate.author,
        "license": normalized_license, "source_license_code": candidate.source_license_code,
        "license_evidence": candidate.license_evidence, "title": candidate.title, "category": candidate.category.value,
        "fetch_date": fetch_date, "width": int(final_image.shape[1]), "height": int(final_image.shape[0]),
        "physical_extent_mm": candidate.physical_extent_mm, "tile_count": processed.tile_count,
        "crop_fraction": round(processed.crop_fraction, 3),
    }, None


def fetch_web_backgrounds(output_directory: Path, sources: list[WebBackgroundSource], max_new_per_source: int | None) -> dict:
    """Run every source in order; returns per-source counts of kept and skipped candidates."""
    background_root = output_directory.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    rejected_directory = background_root / REJECTED_WEB_SUBDIRECTORY
    sources_log, skips_log = output_directory / SOURCES_LOG_NAME, output_directory / SKIPS_LOG_NAME
    already_handled = (read_logged_file_names(sources_log) | read_logged_file_names(skips_log)
                       | {path.name for path in output_directory.glob("*.jpg")}
                       | {path.name for path in rejected_directory.glob("*.jpg")})
    duplicate_index = build_existing_duplicate_index(background_root)
    client, fetch_date = PoliteHttpClient(), datetime.date.today().isoformat()
    counts = {}
    for source in sources:
        candidates = CANDIDATE_LISTERS[source](client)
        logger.info("%s: %d candidates", source.value, len(candidates))
        kept = skipped = 0
        for candidate in candidates:
            if candidate.output_file_name in already_handled:
                continue
            if max_new_per_source is not None and kept + skipped >= max_new_per_source:
                break
            try:
                row, skip_reason = process_candidate(client, candidate, output_directory, duplicate_index, fetch_date)
            except (urllib.error.URLError, ValueError, zipfile.BadZipFile, OSError) as error:
                # One broken download must not end a run over hundreds of independent images.
                row, skip_reason = None, f"download failed: {error}"
            already_handled.add(candidate.output_file_name)
            if row is not None:
                append_json_line(sources_log, row)
                kept += 1
                logger.info("kept %s (%s)", candidate.output_file_name, candidate.category.value)
            else:
                append_json_line(skips_log, {"file": candidate.output_file_name, "reason": skip_reason,
                                             "image_url": candidate.image_url, "date": fetch_date})
                skipped += 1
                logger.info("skip %s: %s", candidate.output_file_name, skip_reason)
        counts[source.value] = {"kept": kept, "skipped": skipped}
    return counts


def parse_arguments(argument_list: list[str] | None = None) -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=BACKGROUND_DIR / "web", help="the web/ folder under a background root")
    parser.add_argument("--sources", nargs="+", default=[source.value for source in WebBackgroundSource],
                        choices=[source.value for source in WebBackgroundSource])
    parser.add_argument("--max-new-per-source", type=int, default=None, help="stop each source after this many new candidates")
    return parser.parse_args(argument_list)


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    arguments = parse_arguments()
    counts = fetch_web_backgrounds(arguments.output, [WebBackgroundSource(name) for name in arguments.sources],
                                   arguments.max_new_per_source)
    logger.info("done: %s", json.dumps(counts))


if __name__ == "__main__":
    main()
