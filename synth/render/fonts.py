"""Font registry: which font files the renderer may use, their role, and their license.

The font files themselves are NOT in the repo (see `fetch_fonts.py`); they live under
`paths.FONT_DIR`. Every entry records its license so the README table stays honest.

- PRINTED fonts draw pre-printed check text and computer-printed fill-ins.
- HANDWRITING fonts imitate a pen for handwritten fill-ins (payee, date, amounts, memo).
- SIGNATURE fonts are connected scripts used only for the signature.
- MICR is GnuMICR, a GPL-2 E-13B font. Only its rendered pixels leave this machine.
"""

import functools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PIL import ImageFont

from synth.paths import FONT_DIR


class FontRole(str, Enum):
    """What a font is allowed to draw on a check."""

    PRINTED = "printed"
    HANDWRITING = "handwriting"
    SIGNATURE = "signature"
    MICR = "micr"


@dataclass(frozen=True)
class FontSpec:
    """One font file the renderer can load."""

    font_id: str
    relative_path: str
    role: FontRole
    license_name: str
    source_url: str


GOOGLE_FONTS_RAW = "https://raw.githubusercontent.com/google/fonts/main/ofl"

FONT_SPECS: list[FontSpec] = [
    FontSpec("libre_baskerville", "librebaskerville/LibreBaskerville-Variable.ttf", FontRole.PRINTED, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/librebaskerville/LibreBaskerville%5Bwght%5D.ttf"),
    FontSpec("eb_garamond", "ebgaramond/EBGaramond-Variable.ttf", FontRole.PRINTED, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/ebgaramond/EBGaramond%5Bwght%5D.ttf"),
    FontSpec("source_sans_3", "sourcesans3/SourceSans3-Variable.ttf", FontRole.PRINTED, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/sourcesans3/SourceSans3%5Bwght%5D.ttf"),
    FontSpec("pt_sans", "ptsans/PT_Sans-Web-Regular.ttf", FontRole.PRINTED, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/ptsans/PT_Sans-Web-Regular.ttf"),
    FontSpec("pt_sans_bold", "ptsans/PT_Sans-Web-Bold.ttf", FontRole.PRINTED, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/ptsans/PT_Sans-Web-Bold.ttf"),
    FontSpec("oswald", "oswald/Oswald-Variable.ttf", FontRole.PRINTED, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/oswald/Oswald%5Bwght%5D.ttf"),
    FontSpec("courier_prime", "courierprime/CourierPrime-Regular.ttf", FontRole.PRINTED, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/courierprime/CourierPrime-Regular.ttf"),
    FontSpec("courier_prime_bold", "courierprime/CourierPrime-Bold.ttf", FontRole.PRINTED, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/courierprime/CourierPrime-Bold.ttf"),
    FontSpec("caveat", "caveat/Caveat-Variable.ttf", FontRole.HANDWRITING, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/caveat/Caveat%5Bwght%5D.ttf"),
    FontSpec("kalam", "kalam/Kalam-Regular.ttf", FontRole.HANDWRITING, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/kalam/Kalam-Regular.ttf"),
    FontSpec("nothing_you_could_do", "nothingyoucoulddo/NothingYouCouldDo.ttf", FontRole.HANDWRITING, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/nothingyoucoulddo/NothingYouCouldDo.ttf"),
    FontSpec("reenie_beanie", "reeniebeanie/ReenieBeanie.ttf", FontRole.HANDWRITING, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/reeniebeanie/ReenieBeanie.ttf"),
    FontSpec("shadows_into_light", "shadowsintolight/ShadowsIntoLight.ttf", FontRole.HANDWRITING, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/shadowsintolight/ShadowsIntoLight.ttf"),
    FontSpec("indie_flower", "indieflower/IndieFlower-Regular.ttf", FontRole.HANDWRITING, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/indieflower/IndieFlower-Regular.ttf"),
    FontSpec("dancing_script", "dancingscript/DancingScript-Variable.ttf", FontRole.SIGNATURE, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/dancingscript/DancingScript%5Bwght%5D.ttf"),
    FontSpec("mr_dafoe", "mrdafoe/MrDafoe-Regular.ttf", FontRole.SIGNATURE, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/mrdafoe/MrDafoe-Regular.ttf"),
    FontSpec("allura", "allura/Allura-Regular.ttf", FontRole.SIGNATURE, "OFL-1.1", f"{GOOGLE_FONTS_RAW}/allura/Allura-Regular.ttf"),
    FontSpec("gnu_micr", "gnumicr/GnuMICR.ttf", FontRole.MICR, "GPL-2.0 (font only; never bundled)", "https://raw.githubusercontent.com/alerque/gnumicr/master/GnuMICR.ttf"),
]

FONT_SPECS_BY_ID: dict[str, FontSpec] = {spec.font_id: spec for spec in FONT_SPECS}

# GnuMICR maps the four E-13B control symbols onto capital letters.
MICR_TRANSIT_SYMBOL = "A"
MICR_AMOUNT_SYMBOL = "B"
MICR_ON_US_SYMBOL = "C"
MICR_DASH_SYMBOL = "D"


def font_ids_with_role(role: FontRole) -> list[str]:
    """Return every registered font id with the given role, in registry order."""
    return [spec.font_id for spec in FONT_SPECS if spec.role == role]


def font_file_path(font_id: str, font_dir: Path = FONT_DIR) -> Path:
    """Resolve a font id to its file on disk, failing loudly if it was never fetched."""
    font_path = font_dir / FONT_SPECS_BY_ID[font_id].relative_path
    if not font_path.exists():
        raise FileNotFoundError(f"font {font_id} missing at {font_path}; run `python -m synth.fetch_fonts`")
    return font_path


@functools.lru_cache(maxsize=512)
def load_font(font_id: str, pixel_size: int) -> ImageFont.FreeTypeFont:
    """Load (and cache per process) a font at a pixel size.

    Variable fonts default to their first instance (often ExtraLight), so pin them to Regular.
    """
    font = ImageFont.truetype(str(font_file_path(font_id)), max(4, int(pixel_size)))
    if "Variable" in FONT_SPECS_BY_ID[font_id].relative_path:
        font.set_variation_by_name("Regular")
    return font
