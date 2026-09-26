"""One downloadable web background and the license gate every candidate must pass.

Web backgrounds train a model that ships publicly, so only CC0 and public-domain images are
allowed, and the license must come from the source's own metadata:

- Poly Haven and ambientCG publish CC0 for their entire catalogue (no per-asset field exists),
  so every asset they list is CC0; `license_evidence` points at the site-wide license page.
- Openverse reports a `license` code per result; Wikimedia Commons reports `extmetadata.License`
  per file. Only the codes in `ACCEPTED_SOURCE_LICENSE_CODES` pass; CC BY, CC BY-SA, "no known
  restrictions" and anything unknown are rejected.
"""

from dataclasses import dataclass
from enum import Enum


class WebBackgroundSource(str, Enum):
    """The keyless sources the fetcher pulls from."""

    POLYHAVEN = "polyhaven"
    AMBIENTCG = "ambientcg"
    OPENVERSE = "openverse"
    COMMONS = "commons"


class SurfaceCategory(str, Enum):
    """What kind of surface a background shows; `is_soft_textile` groups the bedding/fabric half."""

    BEDDING = "bedding"             # sheets, duvets, blankets, quilts
    TABLE_LINEN = "table_linen"     # tablecloths
    FABRIC = "fabric"               # upholstery and other woven / knitted cloth
    RUG_CARPET = "rug_carpet"
    LEATHER = "leather"
    WOOD_TABLE = "wood_table"       # fine-grain table tops, veneers, desks
    WOOD_FLOOR = "wood_floor"       # planks, parquet, laminate
    STONE_COUNTER = "stone_counter" # marble, granite, terrazzo, laminate counters
    FLOOR_TILE = "floor_tile"       # tiles, linoleum
    CARDBOARD = "cardboard"
    DESK_MAT = "desk_mat"           # cork, particle board, desk surfaces

    @property
    def is_soft_textile(self) -> bool:
        """True for the bedding-and-fabric half of the target mix."""
        return self in (SurfaceCategory.BEDDING, SurfaceCategory.TABLE_LINEN,
                        SurfaceCategory.FABRIC, SurfaceCategory.RUG_CARPET)


# Normalized license names written to SOURCES.jsonl.
CC0_LICENSE = "CC0-1.0"
PUBLIC_DOMAIN_MARK = "PDM-1.0"
PUBLIC_DOMAIN = "public-domain"

# (source, license code as the source reports it) -> normalized license. Anything absent is rejected.
ACCEPTED_SOURCE_LICENSE_CODES: dict[tuple[WebBackgroundSource, str], str] = {
    (WebBackgroundSource.POLYHAVEN, "cc0"): CC0_LICENSE,
    (WebBackgroundSource.AMBIENTCG, "cc0"): CC0_LICENSE,
    (WebBackgroundSource.OPENVERSE, "cc0"): CC0_LICENSE,
    (WebBackgroundSource.OPENVERSE, "pdm"): PUBLIC_DOMAIN_MARK,
    (WebBackgroundSource.COMMONS, "cc0"): CC0_LICENSE,
    (WebBackgroundSource.COMMONS, "pd"): PUBLIC_DOMAIN,
}


@dataclass(frozen=True)
class WebBackgroundCandidate:
    """Everything needed to download one image, judge its license and credit it.

    - `source_license_code`: the license exactly as the source's metadata reported it (lowercased).
    - `license_evidence`: where that license came from (API field name or site-wide policy URL).
    - `physical_extent_mm`: real-world width of the full image for scanned textures (None for photos
      or when unknown); drives tiling / cropping to a common scale.
    - `is_seamless_texture`: tiles without seams, so small textures may be repeated 2x2.
    """

    source: WebBackgroundSource
    source_id: str
    slug: str
    title: str
    author: str
    source_license_code: str
    license_evidence: str
    source_page_url: str
    image_url: str
    category: SurfaceCategory
    physical_extent_mm: float | None = None
    is_seamless_texture: bool = False
    reported_width: int | None = None
    reported_height: int | None = None

    @property
    def output_file_name(self) -> str:
        """`web__<source>__<slug>__<id>.jpg`, the stable name used for resume and dedupe logs."""
        return f"web__{self.source.value}__{self.slug}__{self.source_id}.jpg"


def accepted_license(candidate: WebBackgroundCandidate) -> str | None:
    """The normalized license if the candidate's reported license is CC0 / public domain, else None."""
    return ACCEPTED_SOURCE_LICENSE_CODES.get((candidate.source, candidate.source_license_code.strip().lower()))


def slugify_title(title: str, max_length: int = 40) -> str:
    """Lowercase ASCII words joined by hyphens, cut at a word boundary; never empty, never contains `__`."""
    words, current = [], ""
    for character in title.lower():
        if character.isascii() and character.isalnum():
            current += character
        elif current:
            words.append(current)
            current = ""
    if current:
        words.append(current)
    slug = ""
    for word in words:
        if len(slug) + len(word) + 1 > max_length:
            break
        slug = f"{slug}-{word}" if slug else word
    return slug or "untitled"
