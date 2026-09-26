"""ambientCG materials (CC0, site-wide): select check-surface materials, pull only the Color map.

API: `https://ambientcg.com/api/v2/full_json?type=Material&include=downloadData,tagData,dimensionsData`
paged with `limit`/`offset`. Each material ships as a zip of maps; `download_ambientcg_color_map`
reads the zip's central directory and the single `*_Color.jpg` member over HTTP Range requests,
so no archive is ever written to disk.

- Photo-based scans (photogrammetry / multi-angle / approximated) are taken in every selected category.
- Substance-procedural materials look synthetic more often, so they are only taken for materials
  that have no scan (marble, granite, terrazzo, cardboard, chipboard), capped per category by popularity.
- `dimensionX` is in centimetres; 0 means unknown.
"""

import io
import zipfile

from synthetic_backgrounds.web_sources.web_candidate import SurfaceCategory, WebBackgroundCandidate, WebBackgroundSource
from synthetic_backgrounds.web_sources.web_http import HttpRangeFile, PoliteHttpClient

API_URL = "https://ambientcg.com/api/v2/full_json"
LICENSE_POLICY_URL = "https://docs.ambientcg.com/license/"
PAGE_SIZE = 250
ZIP_ATTRIBUTE = "2K-JPG"
RANGE_READ_BUFFER_BYTES = 1 << 20
PROCEDURAL_METHOD = "PBRProcedural"
PROCEDURAL_CAP_PER_CATEGORY = 8

DISPLAY_CATEGORY_TO_SURFACE: dict[str, SurfaceCategory] = {
    "Fabric": SurfaceCategory.FABRIC,
    "Carpet": SurfaceCategory.RUG_CARPET,
    "Leather": SurfaceCategory.LEATHER,
    "Wood": SurfaceCategory.WOOD_TABLE,
    "Wood Floor": SurfaceCategory.WOOD_FLOOR,
    "Marble": SurfaceCategory.STONE_COUNTER,
    "Granite": SurfaceCategory.STONE_COUNTER,
    "Terrazzo": SurfaceCategory.STONE_COUNTER,
    "Tiles": SurfaceCategory.FLOOR_TILE,
    "Cork": SurfaceCategory.DESK_MAT,
    "Chipboard": SurfaceCategory.DESK_MAT,
    "Cardboard": SurfaceCategory.CARDBOARD,
}
PROCEDURAL_ALLOWED_CATEGORIES = {"Marble", "Granite", "Terrazzo", "Cardboard", "Chipboard"}
EXCLUDED_TAGS = {"bark", "log", "logs", "firewood", "branch", "fence", "siding", "roof", "wall", "facade",
                 "chainmail", "plastisol", "printed", "damaged", "carbon", "outdoor", "pavement", "paving",
                 "sidewalk", "brick", "bricks", "moss", "dirty", "rotten", "burned"}


def fetch_ambientcg_catalogue(client: PoliteHttpClient) -> list[dict]:
    """Every material in the catalogue (a few pages)."""
    assets, offset = [], 0
    while True:
        page = client.get_json(API_URL, {"type": "Material", "include": "downloadData,tagData,dimensionsData",
                                         "limit": PAGE_SIZE, "offset": offset, "sort": "Alphabet"})
        assets.extend(page["foundAssets"])
        offset += PAGE_SIZE
        if not page["foundAssets"] or offset >= page["numberOfResults"]:
            return assets


def zip_download_url(asset: dict, attribute: str = ZIP_ATTRIBUTE) -> str | None:
    """URL of the asset's zip with this resolution attribute, if offered."""
    for folder in asset.get("downloadFolders", {}).values():
        for download in folder.get("downloadFiletypeCategories", {}).get("zip", {}).get("downloads", []):
            if download.get("attribute") == attribute:
                return download["fullDownloadPath"]
    return None


def select_ambientcg_assets(assets: list[dict]) -> list[tuple[dict, SurfaceCategory]]:
    """Filter the catalogue to check surfaces, applying the procedural policy; sorted by asset id."""
    selected, procedural_by_category = [], {}
    for asset in assets:
        display_category = asset.get("displayCategory")
        surface = DISPLAY_CATEGORY_TO_SURFACE.get(display_category)
        if surface is None or EXCLUDED_TAGS & {tag.lower() for tag in asset.get("tags", [])}:
            continue
        if asset.get("creationMethod") == PROCEDURAL_METHOD:
            if display_category in PROCEDURAL_ALLOWED_CATEGORIES:
                procedural_by_category.setdefault(display_category, []).append((asset, surface))
            continue
        selected.append((asset, surface))
    for category_assets in procedural_by_category.values():
        category_assets.sort(key=lambda pair: (-pair[0].get("popularityScore", 0), pair[0]["assetId"]))
        selected.extend(category_assets[:PROCEDURAL_CAP_PER_CATEGORY])
    return sorted(selected, key=lambda pair: pair[0]["assetId"])


def list_ambientcg_candidates(client: PoliteHttpClient) -> list[WebBackgroundCandidate]:
    """Candidates for every selected material that offers a 2K JPG zip."""
    candidates = []
    for asset, surface in select_ambientcg_assets(fetch_ambientcg_catalogue(client)):
        zip_url = zip_download_url(asset)
        if zip_url is None:
            continue
        extent_cm = float(max(asset.get("dimensionX") or 0, asset.get("dimensionY") or 0))
        candidates.append(WebBackgroundCandidate(
            source=WebBackgroundSource.AMBIENTCG, source_id=asset["assetId"],
            slug=f"{asset.get('displayCategory', 'material')}-{asset.get('creationMethod', '')}".lower().replace(" ", "-")[:40],
            title=asset.get("displayName") or asset["assetId"], author="ambientCG (Lennart Demes)",
            source_license_code="cc0", license_evidence=LICENSE_POLICY_URL,
            source_page_url=asset.get("shortLink") or f"https://ambientcg.com/view?id={asset['assetId']}",
            image_url=zip_url, category=surface,
            physical_extent_mm=extent_cm * 10 if extent_cm else None, is_seamless_texture=True,
        ))
    return candidates


def download_ambientcg_color_map(client: PoliteHttpClient, candidate: WebBackgroundCandidate) -> bytes:
    """Bytes of the `*_Color.jpg` member of the candidate's remote zip, read by byte ranges."""
    with io.BufferedReader(HttpRangeFile(client, candidate.image_url), RANGE_READ_BUFFER_BYTES) as remote_zip:
        with zipfile.ZipFile(remote_zip) as archive:
            color_members = [name for name in archive.namelist() if name.lower().endswith("_color.jpg")]
            if not color_members:
                raise ValueError(f"no *_Color.jpg in {candidate.image_url}")
            return archive.read(color_members[0])
