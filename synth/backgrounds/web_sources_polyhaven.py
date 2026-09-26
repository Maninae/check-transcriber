"""Poly Haven textures (CC0, site-wide): list surfaces checks get laid on, download the diffuse JPG.

API: `https://api.polyhaven.com/assets?t=textures` (id -> categories, dimensions in mm, authors)
and `/files/<id>` (per-map, per-resolution URLs). Selection is by category plus name rules that
keep fabrics, leather, carpet, fine-grain wood, wood floors, stone counters and indoor tiles and
drop walls, bark, outdoor ground, doors and the like.
"""

import re

from synth.backgrounds.web_candidate import SurfaceCategory, WebBackgroundCandidate, WebBackgroundSource
from synth.backgrounds.web_http import PoliteHttpClient
from synth.backgrounds.web_image_processing import MIN_SHORT_SIDE_PX, TARGET_PHYSICAL_EXTENT_MM

ASSETS_URL = "https://api.polyhaven.com/assets"
FILES_URL = "https://api.polyhaven.com/files"
LICENSE_POLICY_URL = "https://polyhaven.com/license"
WOOD_EXCLUDED_CATEGORIES = {"wall", "outdoor", "bark", "terrain", "aerial", "roofing", "natural"}
WOOD_EXCLUDED_NAME = re.compile(r"barrel|gate|door|shutter|garage|chips|trunk|cabinet|panel|siding|peeling|moss|deck|pathway|rough|medieval|old_|worn|dirt")
STONE_NAME = re.compile(r"marble_0|marble_mosaic|granite_tile|terrazzo|linoleum|slate_floor|floor_tiles_06|tiled_floor|slab_tiles")
FABRIC_EXCLUDED_NAME = re.compile(r"curly_teddy|faux_fur")
PREFERRED_RESOLUTIONS = ("2k", "4k")


def classify_polyhaven_texture(asset_id: str, categories: set[str]) -> SurfaceCategory | None:
    """Surface category for a Poly Haven texture, or None when it is not a plausible check surface."""
    if categories & {"fabric", "leather", "carpet"}:
        if FABRIC_EXCLUDED_NAME.search(asset_id):
            return None
        if "leather" in categories:
            return SurfaceCategory.LEATHER
        return SurfaceCategory.RUG_CARPET if "carpet" in categories else SurfaceCategory.FABRIC
    if STONE_NAME.search(asset_id) and "outdoor" not in categories:
        return SurfaceCategory.FLOOR_TILE if re.search(r"tile|linoleum|slate", asset_id) else SurfaceCategory.STONE_COUNTER
    if "wood" in categories and not categories & WOOD_EXCLUDED_CATEGORIES and not WOOD_EXCLUDED_NAME.search(asset_id):
        return SurfaceCategory.WOOD_FLOOR if "floor" in categories or "floor" in asset_id else SurfaceCategory.WOOD_TABLE
    return None


def list_polyhaven_candidates(client: PoliteHttpClient) -> list[WebBackgroundCandidate]:
    """Every matching texture, sorted by id, with its diffuse JPG URL at the smallest adequate resolution."""
    assets = client.get_json(ASSETS_URL, {"t": "textures"})
    candidates = []
    for asset_id in sorted(assets):
        asset = assets[asset_id]
        category = classify_polyhaven_texture(asset_id, set(asset.get("categories", [])))
        if category is None:
            continue
        extent_mm = float(max(asset.get("dimensions") or [0]))
        # Large scans are centre-cropped to the target extent, so they need a higher resolution.
        needed_pixels = MIN_SHORT_SIDE_PX * 1.2 * max(1.0, extent_mm / TARGET_PHYSICAL_EXTENT_MM)
        diffuse_urls = client.get_json(f"{FILES_URL}/{asset_id}").get("Diffuse", {})
        available = [resolution for resolution in PREFERRED_RESOLUTIONS if "jpg" in diffuse_urls.get(resolution, {})]
        if not available:
            continue
        resolution = next((r for r in available if int(r[:-1]) * 1024 >= needed_pixels), available[-1])
        candidates.append(WebBackgroundCandidate(
            source=WebBackgroundSource.POLYHAVEN, source_id=asset_id, slug=asset_id.replace("_", "-")[:40],
            title=asset.get("name", asset_id), author=", ".join(sorted(asset.get("authors", {}))),
            source_license_code="cc0", license_evidence=LICENSE_POLICY_URL,
            source_page_url=f"https://polyhaven.com/a/{asset_id}",
            image_url=diffuse_urls[resolution]["jpg"]["url"], category=category,
            physical_extent_mm=extent_mm or None, is_seamless_texture=True,
        ))
    return candidates
