"""Font registry: which font files the renderer may use, their role, and their license.

The font files themselves are NOT in the repo (see `fetch_fonts.py`); they live under
`paths.FONT_DIR`. Every entry records its license so the README table stays honest.

- PRINTED fonts draw pre-printed check text and computer-printed fill-ins.
- HANDWRITING fonts supply letter shapes for handwritten fill-ins (payee, date, amounts, memo);
  the pen engine reduces them to a centreline and re-inks them, so stroke weight does not matter.
- SIGNATURE fonts are connected scripts used only for the signature.
- Handwriting and signature fonts are SIL OFL 1.1 or Apache 2.0 only, from google/fonts.
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
    """One font file the renderer can load.

    - `is_cursive`: letters join, so the handwriting engine renders whole words (joins survive)
      instead of placing glyphs one by one.
    - `is_variable`: a variable font, pinned to its Regular instance on load.
    """

    font_id: str
    relative_path: str
    role: FontRole
    license_name: str
    source_url: str
    is_cursive: bool = False
    is_variable: bool = False


GOOGLE_FONTS_REPO_RAW = "https://raw.githubusercontent.com/google/fonts/main"
GOOGLE_FONTS_RAW = f"{GOOGLE_FONTS_REPO_RAW}/ofl"
OFL_LICENSE = "OFL-1.1"
APACHE_LICENSE = "Apache-2.0"
LICENSE_DIRECTORY_BY_NAME = {OFL_LICENSE: "ofl", APACHE_LICENSE: "apache"}


def google_font(font_id: str, directory: str, filename: str, role: FontRole, license_name: str = OFL_LICENSE,
                is_cursive: bool = False) -> FontSpec:
    """Registry entry for a file in the google/fonts repo, stored as `<directory>/<filename>` on the data drive.

    - `license_name` picks the upstream folder: `ofl/` (SIL OFL 1.1) or `apache/` (Apache 2.0).
    - Upstream variable-font names contain `[wght]`, which is URL-escaped here.
    """
    license_directory = LICENSE_DIRECTORY_BY_NAME[license_name]
    escaped_filename = filename.replace("[", "%5B").replace("]", "%5D")
    return FontSpec(font_id, f"{directory}/{filename}", role, license_name,
                    f"{GOOGLE_FONTS_REPO_RAW}/{license_directory}/{directory}/{escaped_filename}",
                    is_cursive=is_cursive, is_variable="[" in filename)


HANDWRITING = FontRole.HANDWRITING
SIGNATURE = FontRole.SIGNATURE

FONT_SPECS: list[FontSpec] = [
    FontSpec("libre_baskerville", "librebaskerville/LibreBaskerville-Variable.ttf", FontRole.PRINTED, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/librebaskerville/LibreBaskerville%5Bwght%5D.ttf", is_variable=True),
    FontSpec("eb_garamond", "ebgaramond/EBGaramond-Variable.ttf", FontRole.PRINTED, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/ebgaramond/EBGaramond%5Bwght%5D.ttf", is_variable=True),
    FontSpec("source_sans_3", "sourcesans3/SourceSans3-Variable.ttf", FontRole.PRINTED, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/sourcesans3/SourceSans3%5Bwght%5D.ttf", is_variable=True),
    FontSpec("pt_sans", "ptsans/PT_Sans-Web-Regular.ttf", FontRole.PRINTED, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/ptsans/PT_Sans-Web-Regular.ttf"),
    FontSpec("pt_sans_bold", "ptsans/PT_Sans-Web-Bold.ttf", FontRole.PRINTED, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/ptsans/PT_Sans-Web-Bold.ttf"),
    FontSpec("oswald", "oswald/Oswald-Variable.ttf", FontRole.PRINTED, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/oswald/Oswald%5Bwght%5D.ttf", is_variable=True),
    FontSpec("courier_prime", "courierprime/CourierPrime-Regular.ttf", FontRole.PRINTED, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/courierprime/CourierPrime-Regular.ttf"),
    FontSpec("courier_prime_bold", "courierprime/CourierPrime-Bold.ttf", FontRole.PRINTED, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/courierprime/CourierPrime-Bold.ttf"),
    # Handwriting: print hands first, then joined (cursive) hands. Rejected by eye at a real pen
    # width (letters clog): Just Another Hand, Loved by the King, Homemade Apple, Waiting for
    # the Sunrise, Cedarville Cursive, Over the Rainbow.
    FontSpec("caveat", "caveat/Caveat-Variable.ttf", HANDWRITING, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/caveat/Caveat%5Bwght%5D.ttf", is_variable=True),
    FontSpec("kalam", "kalam/Kalam-Regular.ttf", HANDWRITING, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/kalam/Kalam-Regular.ttf"),
    FontSpec("nothing_you_could_do", "nothingyoucoulddo/NothingYouCouldDo.ttf", HANDWRITING, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/nothingyoucoulddo/NothingYouCouldDo.ttf"),
    FontSpec("reenie_beanie", "reeniebeanie/ReenieBeanie.ttf", HANDWRITING, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/reeniebeanie/ReenieBeanie.ttf"),
    FontSpec("shadows_into_light", "shadowsintolight/ShadowsIntoLight.ttf", HANDWRITING, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/shadowsintolight/ShadowsIntoLight.ttf"),
    FontSpec("indie_flower", "indieflower/IndieFlower-Regular.ttf", HANDWRITING, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/indieflower/IndieFlower-Regular.ttf"),
    google_font("patrick_hand", "patrickhand", "PatrickHand-Regular.ttf", HANDWRITING),
    google_font("gochi_hand", "gochihand", "GochiHand-Regular.ttf", HANDWRITING),
    google_font("covered_by_your_grace", "coveredbyyourgrace", "CoveredByYourGrace.ttf", HANDWRITING),
    google_font("nanum_pen_script", "nanumpenscript", "NanumPenScript-Regular.ttf", HANDWRITING),
    google_font("gaegu", "gaegu", "Gaegu-Regular.ttf", HANDWRITING),
    google_font("architects_daughter", "architectsdaughter", "ArchitectsDaughter-Regular.ttf", HANDWRITING),
    google_font("schoolbell", "schoolbell", "Schoolbell-Regular.ttf", HANDWRITING, APACHE_LICENSE),
    google_font("coming_soon", "comingsoon", "ComingSoon-Regular.ttf", HANDWRITING, APACHE_LICENSE),
    google_font("handlee", "handlee", "Handlee-Regular.ttf", HANDWRITING),
    google_font("neucha", "neucha", "Neucha.ttf", HANDWRITING),
    google_font("sue_ellen_francisco", "sueellenfrancisco", "SueEllenFrancisco-Regular.ttf", HANDWRITING),
    google_font("annie_use_your_telescope", "annieuseyourtelescope", "AnnieUseYourTelescope-Regular.ttf", HANDWRITING),
    google_font("just_me_again_down_here", "justmeagaindownhere", "JustMeAgainDownHere.ttf", HANDWRITING),
    google_font("mynerve", "mynerve", "Mynerve-Regular.ttf", HANDWRITING),
    google_font("edu_sa_beginner", "edusabeginner", "EduSABeginner[wght].ttf", HANDWRITING),
    google_font("edu_nsw_act_foundation", "edunswactfoundation", "EduNSWACTFoundation[wght].ttf", HANDWRITING),
    google_font("edu_vic_wa_nt_beginner", "eduvicwantbeginner", "EduVICWANTBeginner[wght].ttf", HANDWRITING),
    google_font("edu_qld_beginner", "eduqldbeginner", "EduQLDBeginner[wght].ttf", HANDWRITING),
    google_font("the_girl_next_door", "thegirlnextdoor", "TheGirlNextDoor.ttf", HANDWRITING),
    google_font("give_you_glory", "giveyouglory", "GiveYouGlory.ttf", HANDWRITING),
    google_font("grape_nuts", "grapenuts", "GrapeNuts-Regular.ttf", HANDWRITING),
    google_font("short_stack", "shortstack", "ShortStack-Regular.ttf", HANDWRITING),
    google_font("beth_ellen", "bethellen", "BethEllen-Regular.ttf", HANDWRITING, is_cursive=True),
    google_font("la_belle_aurore", "labelleaurore", "LaBelleAurore.ttf", HANDWRITING, is_cursive=True),
    google_font("dawning_of_a_new_day", "dawningofanewday", "DawningofaNewDay.ttf", HANDWRITING, is_cursive=True),
    google_font("zeyada", "zeyada", "Zeyada.ttf", HANDWRITING, is_cursive=True),
    google_font("edu_tas_beginner", "edutasbeginner", "EduTASBeginner[wght].ttf", HANDWRITING, is_cursive=True),
    google_font("marck_script", "marckscript", "MarckScript-Regular.ttf", HANDWRITING, is_cursive=True),
    # Signatures: all joined scripts. Rejected by eye (x-height too small for a real pen, they
    # collapse into tangles): Herr Von Muellerhoff, Monsieur La Doulaise.
    FontSpec("dancing_script", "dancingscript/DancingScript-Variable.ttf", SIGNATURE, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/dancingscript/DancingScript%5Bwght%5D.ttf", is_cursive=True, is_variable=True),
    FontSpec("mr_dafoe", "mrdafoe/MrDafoe-Regular.ttf", SIGNATURE, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/mrdafoe/MrDafoe-Regular.ttf", is_cursive=True),
    FontSpec("allura", "allura/Allura-Regular.ttf", SIGNATURE, OFL_LICENSE, f"{GOOGLE_FONTS_RAW}/allura/Allura-Regular.ttf", is_cursive=True),
    google_font("mrs_saint_delafield", "mrssaintdelafield", "MrsSaintDelafield-Regular.ttf", SIGNATURE, is_cursive=True),
    google_font("kristi", "kristi", "Kristi-Regular.ttf", SIGNATURE, is_cursive=True),
    google_font("sacramento", "sacramento", "Sacramento-Regular.ttf", SIGNATURE, is_cursive=True),
    google_font("ruthie", "ruthie", "Ruthie-Regular.ttf", SIGNATURE, is_cursive=True),
    google_font("qwigley", "qwigley", "Qwigley-Regular.ttf", SIGNATURE, is_cursive=True),
    google_font("yellowtail", "yellowtail", "Yellowtail-Regular.ttf", SIGNATURE, APACHE_LICENSE, is_cursive=True),
    google_font("meddon", "meddon", "Meddon.ttf", SIGNATURE, is_cursive=True),
    google_font("arizonia", "arizonia", "Arizonia-Regular.ttf", SIGNATURE, is_cursive=True),
    google_font("whisper", "whisper", "Whisper-Regular.ttf", SIGNATURE, is_cursive=True),
    google_font("ephesis", "ephesis", "Ephesis-Regular.ttf", SIGNATURE, is_cursive=True),
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
        raise FileNotFoundError(f"font {font_id} missing at {font_path}; run `python -m synth.render.fetch_fonts`")
    return font_path


@functools.lru_cache(maxsize=512)
def load_font(font_id: str, pixel_size: int) -> ImageFont.FreeTypeFont:
    """Load (and cache per process) a font at a pixel size.

    Variable fonts default to their first instance (often ExtraLight), so pin them to Regular.
    """
    font = ImageFont.truetype(str(font_file_path(font_id)), max(4, int(pixel_size)))
    if FONT_SPECS_BY_ID[font_id].is_variable:
        font.set_variation_by_name("Regular")
    return font
