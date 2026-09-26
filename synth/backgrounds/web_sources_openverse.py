"""Openverse image search (keyless), restricted to CC0 and Public Domain Mark at query time and re-checked per result.

`https://api.openverse.org/v1/images/?q=<term>&license=cc0,pdm&page_size=20`. Anonymous clients
get ~20 requests a minute and ~200 a day, so the fetcher spaces requests (see `web_http`) and
reads only `PAGES_PER_TERM` pages per term. `size=large` leaves mostly full-resolution StockSnap
photos (small Flickr copies fail the 900 px floor anyway). Wikimedia results are excluded here
because Commons is searched directly with its own per-file license metadata.
"""

import logging
import urllib.error

from synth.backgrounds.web_candidate import SurfaceCategory, WebBackgroundCandidate, WebBackgroundSource, slugify_title
from synth.backgrounds.web_http import PoliteHttpClient
from synth.backgrounds.web_image_processing import MAX_ASPECT_RATIO, MIN_SHORT_SIDE_PX
from synth.backgrounds.web_photo_search_terms import PHOTO_SEARCH_TERMS, is_excluded_title

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.openverse.org/v1/images/"
PAGE_SIZE = 20
PAGES_PER_TERM = 3
EXCLUDED_PROVIDERS = "wikimedia"


def openverse_result_to_candidate(result: dict, category: SurfaceCategory) -> WebBackgroundCandidate | None:
    """A candidate from one search result, or None when metadata already rules it out (size, aspect, title)."""
    width, height = result.get("width") or 0, result.get("height") or 0
    if width and height and (min(width, height) < MIN_SHORT_SIDE_PX or max(width, height) / min(width, height) > MAX_ASPECT_RATIO):
        return None
    title = result.get("title") or ""
    if is_excluded_title(title) or not result.get("url"):
        return None
    return WebBackgroundCandidate(
        source=WebBackgroundSource.OPENVERSE, source_id=result["id"].split("-")[0], slug=slugify_title(title),
        title=title, author=result.get("creator") or "unknown",
        source_license_code=result.get("license", ""), license_evidence="openverse result field `license`",
        source_page_url=result.get("foreign_landing_url") or result.get("detail_url", ""),
        image_url=result["url"], category=category,
        reported_width=width or None, reported_height=height or None,
    )


def list_openverse_candidates(client: PoliteHttpClient) -> list[WebBackgroundCandidate]:
    """Candidates for every search term, in term order then result order, deduplicated by Openverse id."""
    candidates, seen_ids = [], set()
    for term, category in PHOTO_SEARCH_TERMS:
        for page_number in range(1, PAGES_PER_TERM + 1):
            try:
                page = client.get_json(SEARCH_URL, {"q": term, "license": "cc0,pdm", "page_size": PAGE_SIZE,
                                                    "page": page_number, "excluded_source": EXCLUDED_PROVIDERS,
                                                    "size": "large", "mature": "false"})
            except urllib.error.HTTPError as http_error:
                logger.warning("openverse %r page %d failed: HTTP %d", term, page_number, http_error.code)
                break
            for result in page.get("results", []):
                candidate = openverse_result_to_candidate(result, category)
                if candidate is not None and result["id"] not in seen_ids:
                    seen_ids.add(result["id"])
                    candidates.append(candidate)
            if page_number >= page.get("page_count", 0):
                break
    return candidates
