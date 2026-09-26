"""Wikimedia Commons photos whose file page states CC0 or public domain.

Two kinds of query, restricted by structured data:
- the shared photo search terms (full text, first `RESULTS_PER_PAGE` hits each), CC0 statement
  (P275=Q6938433) only: full-text hits among public-domain files are almost all old art and scans;
- `COMMONS_CATEGORY_QUERIES`, texture categories paged deeper (`deepcat:` includes subcategories),
  CC0 or public-domain status (P6216=Q19652).
The search filter only finds candidates: the gate reads each file's own `extmetadata.License`
(`cc0` / `pd`). Downloads use Commons' 2048 px thumbnail rather than multi-megapixel originals.
"""

from synth.backgrounds.web_candidate import SurfaceCategory, WebBackgroundCandidate, WebBackgroundSource, slugify_title
from synth.backgrounds.web_http import PoliteHttpClient
from synth.backgrounds.web_image_processing import MAX_ASPECT_RATIO, MAX_LONG_EDGE_PX, MIN_SHORT_SIDE_PX
from synth.backgrounds.web_photo_search_terms import PHOTO_SEARCH_TERMS, is_excluded_title

API_URL = "https://commons.wikimedia.org/w/api.php"
RESULTS_PER_PAGE = 50
CC0_STATEMENT = "haswbstatement:P275=Q6938433"
CC0_OR_PUBLIC_DOMAIN_STATEMENT = "haswbstatement:P275=Q6938433|P6216=Q19652"
ACCEPTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
# (search expression, category, pages to read)
COMMONS_CATEGORY_QUERIES: tuple[tuple[str, SurfaceCategory, int], ...] = (
    ("deepcat:Textile_textures", SurfaceCategory.FABRIC, 8),
    ("deepcat:Wood_textures", SurfaceCategory.WOOD_TABLE, 2),
    ("deepcat:Wood_grain", SurfaceCategory.WOOD_TABLE, 1),
    ("deepcat:Countertops", SurfaceCategory.STONE_COUNTER, 1),
    ("deepcat:Duvets", SurfaceCategory.BEDDING, 1),
    ("intitle:duvet", SurfaceCategory.BEDDING, 1),
    ("intitle:texture intitle:marble", SurfaceCategory.STONE_COUNTER, 1),
    ("intitle:texture", SurfaceCategory.FABRIC, 4),
)


def strip_html(text: str) -> str:
    """Plain text from Commons' HTML metadata values (Artist is often a link)."""
    plain, inside_tag = "", False
    for character in text:
        if character == "<":
            inside_tag = True
        elif character == ">":
            inside_tag = False
        elif not inside_tag:
            plain += character
    return " ".join(plain.split())


def commons_page_to_candidate(page: dict, category: SurfaceCategory) -> WebBackgroundCandidate | None:
    """A candidate from one search-generator page, or None when size, aspect, type or title rule it out."""
    image_info = (page.get("imageinfo") or [{}])[0]
    width, height = image_info.get("width", 0), image_info.get("height", 0)
    title = page.get("title", "").removeprefix("File:")
    if (not width or not height or min(width, height) < MIN_SHORT_SIDE_PX
            or max(width, height) / min(width, height) > MAX_ASPECT_RATIO
            or image_info.get("mime") not in ACCEPTED_MIME_TYPES or is_excluded_title(title)):
        return None
    metadata = image_info.get("extmetadata", {})
    license_code = metadata.get("License", {}).get("value", "")
    return WebBackgroundCandidate(
        source=WebBackgroundSource.COMMONS, source_id=str(page["pageid"]),
        slug=slugify_title(title.rsplit(".", 1)[0]), title=title,
        author=strip_html(metadata.get("Artist", {}).get("value", "")) or "unknown",
        source_license_code=license_code, license_evidence="commons extmetadata `License`",
        source_page_url=image_info.get("descriptionurl", ""),
        image_url=image_info.get("thumburl") or image_info["url"], category=category,
        reported_width=width, reported_height=height,
    )


def search_commons_pages(client: PoliteHttpClient, search_expression: str, license_statement: str,
                         page_count: int) -> list[dict]:
    """File pages (with imageinfo) for one search, in relevance order, reading up to `page_count` pages."""
    pages, offset = [], 0
    for _ in range(page_count):
        response = client.get_json(API_URL, {
            "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
            "gsrsearch": f"{search_expression} {license_statement} filetype:bitmap",
            "gsrlimit": RESULTS_PER_PAGE, "gsroffset": offset,
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": MAX_LONG_EDGE_PX,
            "iiextmetadatafilter": "License|LicenseShortName|Artist",
        })
        pages.extend(sorted(response.get("query", {}).get("pages", {}).values(), key=lambda page: page.get("index", 0)))
        offset = response.get("continue", {}).get("gsroffset")
        if offset is None:
            break
    return pages


def list_commons_candidates(client: PoliteHttpClient) -> list[WebBackgroundCandidate]:
    """Candidates for every search term then every category query, unique by page id."""
    queries = ([(term, category, CC0_STATEMENT, 1) for term, category in PHOTO_SEARCH_TERMS]
               + [(expression, category, CC0_OR_PUBLIC_DOMAIN_STATEMENT, pages)
                  for expression, category, pages in COMMONS_CATEGORY_QUERIES])
    candidates, seen_page_ids = [], set()
    for search_expression, category, license_statement, page_count in queries:
        for page in search_commons_pages(client, search_expression, license_statement, page_count):
            candidate = commons_page_to_candidate(page, category)
            if candidate is not None and page["pageid"] not in seen_page_ids:
                seen_page_ids.add(page["pageid"])
                candidates.append(candidate)
    return candidates
