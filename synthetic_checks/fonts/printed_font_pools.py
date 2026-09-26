"""Which printed fonts a template may use for each role, and which fonts are held out for eval-class templates.

Every printed font gets a style and a weight; a font's eligible roles and the check kinds that
may use it follow from those, so adding a font is one `PrintedFontProfile` line (plus its
`google_font(...)` line in `fonts.py`).

Roles (what the template's font draws):
- HEADER: payer name, check number, bank name. Bold faces, display faces, regular serifs.
- BODY: payer address, bank city, fractional routing. Regular weight.
- LABEL: pre-printed wording ("Pay to the Order of", "Memo"). Regular weight, no mono on personal stock.
- FILL: computer-printed fill-ins (payee, amount, date, words). Regular sans, serif, mono, condensed.

Check kinds pick different mixes: personal stock leans serif and classic display, business stock
leans sans, condensed and mono (accounting software), money orders are sans/condensed/mono.

Hold-out: `HOLDOUT_FONT_IDS` (~24% of printed fonts, at least one per kind x role pool) appear
ONLY on templates whose index % `HOLDOUT_TEMPLATE_MODULUS` == `HOLDOUT_TEMPLATE_REMAINDER`, and
those templates draw from them with probability `HOLDOUT_FONT_PREFERENCE` per role. If the
dataset puts hold-out-class templates in val/eval, eval sees typefaces train never saw.
`SHARED_PRINTED_FONT_IDS` are fixed on every template (serial, microprint) and never held out.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from synthetic_checks.check_layout import CheckSizeKind

HOLDOUT_TEMPLATE_MODULUS = 5
HOLDOUT_TEMPLATE_REMAINDER = 4
HOLDOUT_FONT_PREFERENCE = 0.75

# Drawn on every template regardless of its chosen fonts: the serial (courier_prime, render_check)
# and microprint rules / VOID pantograph (pt_sans_bold, stock_canvas and security_pattern).
SHARED_PRINTED_FONT_IDS = ("courier_prime", "pt_sans_bold")


class PrintedStyle(str, Enum):
    """Broad typeface class; decides which check kinds and roles a font suits."""

    SERIF = "serif"
    SANS = "sans"
    CONDENSED = "condensed"
    MONO = "mono"
    CLASSIC_DISPLAY = "classic_display"  # engraved caps, high-contrast bold serif, flared: personal names
    HEAVY_DISPLAY = "heavy_display"      # black grotesque: business and money-order headers


class PrintedRole(str, Enum):
    """What a template font draws (see module docstring)."""

    HEADER = "header"
    BODY = "body"
    LABEL = "label"
    FILL = "fill"


@dataclass(frozen=True)
class PrintedFontProfile:
    """A printed font's style, weight, and whether it is reserved for hold-out templates."""

    font_id: str
    style: PrintedStyle
    is_bold: bool = False
    is_holdout: bool = False


S, N, C, M = PrintedStyle.SERIF, PrintedStyle.SANS, PrintedStyle.CONDENSED, PrintedStyle.MONO
CLASSIC, HEAVY = PrintedStyle.CLASSIC_DISPLAY, PrintedStyle.HEAVY_DISPLAY

PRINTED_FONT_PROFILES: list[PrintedFontProfile] = [
    PrintedFontProfile("libre_baskerville", S),
    PrintedFontProfile("eb_garamond", S),
    PrintedFontProfile("tinos", S),
    PrintedFontProfile("tinos_bold", S, is_bold=True),
    PrintedFontProfile("crimson_text", S),
    PrintedFontProfile("crimson_text_bold", S, is_bold=True, is_holdout=True),
    PrintedFontProfile("libre_caslon_text", S, is_holdout=True),
    PrintedFontProfile("old_standard", S),
    PrintedFontProfile("pt_serif", S),
    PrintedFontProfile("pt_serif_bold", S, is_bold=True),
    PrintedFontProfile("merriweather", S),
    PrintedFontProfile("arvo", S),
    PrintedFontProfile("source_sans_3", N),
    PrintedFontProfile("pt_sans", N),
    PrintedFontProfile("pt_sans_bold", N, is_bold=True),
    PrintedFontProfile("arimo", N),
    PrintedFontProfile("arimo_bold", N, is_bold=True),
    PrintedFontProfile("carlito", N),
    PrintedFontProfile("carlito_bold", N, is_bold=True),
    PrintedFontProfile("open_sans", N),
    PrintedFontProfile("libre_franklin", N, is_holdout=True),
    PrintedFontProfile("libre_franklin_bold", N, is_bold=True, is_holdout=True),
    PrintedFontProfile("lato", N),
    PrintedFontProfile("lato_bold", N, is_bold=True),
    PrintedFontProfile("istok_web", N),
    PrintedFontProfile("oswald", C, is_bold=True),
    PrintedFontProfile("roboto_condensed", C),
    PrintedFontProfile("archivo_narrow", C, is_holdout=True),
    PrintedFontProfile("pt_sans_narrow", C),
    PrintedFontProfile("pt_sans_narrow_bold", C, is_bold=True),
    PrintedFontProfile("barlow_condensed_semibold", C, is_bold=True, is_holdout=True),
    PrintedFontProfile("courier_prime", M),
    PrintedFontProfile("courier_prime_bold", M, is_bold=True),
    PrintedFontProfile("cousine", M),
    PrintedFontProfile("ibm_plex_mono", M),
    PrintedFontProfile("share_tech_mono", M),
    PrintedFontProfile("anonymous_pro", M, is_holdout=True),
    PrintedFontProfile("cinzel", CLASSIC, is_bold=True),
    PrintedFontProfile("playfair_display_bold", CLASSIC, is_bold=True),
    PrintedFontProfile("marcellus", CLASSIC, is_bold=True, is_holdout=True),
    PrintedFontProfile("archivo_black", HEAVY, is_bold=True, is_holdout=True),
]

PRINTED_FONT_PROFILES_BY_ID: dict[str, PrintedFontProfile] = {profile.font_id: profile for profile in PRINTED_FONT_PROFILES}
HOLDOUT_FONT_IDS: frozenset[str] = frozenset(profile.font_id for profile in PRINTED_FONT_PROFILES if profile.is_holdout)

# Style weights per check kind and role: a template first draws a style, then a font within it,
# so a style's share does not depend on how many fonts it has. Absent style = not allowed.
ROLE_STYLE_WEIGHTS_BY_KIND: dict[CheckSizeKind, dict[PrintedRole, dict[PrintedStyle, float]]] = {
    CheckSizeKind.PERSONAL: {
        PrintedRole.HEADER: {S: 0.45, N: 0.35, CLASSIC: 0.2},
        PrintedRole.BODY: {S: 0.5, N: 0.5},
        PrintedRole.LABEL: {S: 0.45, N: 0.55},
        PrintedRole.FILL: {S: 0.3, N: 0.45, M: 0.25},
    },
    CheckSizeKind.BUSINESS: {
        PrintedRole.HEADER: {S: 0.25, N: 0.35, C: 0.2, M: 0.1, HEAVY: 0.1},
        PrintedRole.BODY: {S: 0.25, N: 0.4, C: 0.2, M: 0.15},
        PrintedRole.LABEL: {N: 0.6, C: 0.4},
        PrintedRole.FILL: {S: 0.2, N: 0.3, C: 0.15, M: 0.35},
    },
    CheckSizeKind.MONEY_ORDER: {
        PrintedRole.HEADER: {N: 0.45, C: 0.35, HEAVY: 0.2},
        PrintedRole.BODY: {N: 0.6, C: 0.4},
        PrintedRole.LABEL: {N: 0.6, C: 0.4},
        PrintedRole.FILL: {N: 0.35, C: 0.25, M: 0.4},
    },
}


def profile_suits_role(profile: PrintedFontProfile, role: PrintedRole) -> bool:
    """Weight rule: headers take bold/display faces or regular serifs; every other role is regular weight."""
    if role == PrintedRole.HEADER:
        return profile.is_bold or profile.style == PrintedStyle.SERIF
    return not profile.is_bold


def printed_font_pool(kind: CheckSizeKind, role: PrintedRole, holdout: bool) -> dict[PrintedStyle, list[str]]:
    """Font ids a template of `kind` may use for `role`, grouped by style (registry order).

    `holdout` restricts to held-out fonts, else to regular ones; styles with no font are omitted.
    """
    pool: dict[PrintedStyle, list[str]] = {}
    for profile in PRINTED_FONT_PROFILES:
        if profile.style in ROLE_STYLE_WEIGHTS_BY_KIND[kind][role] and profile_suits_role(profile, role) \
                and profile.is_holdout == holdout:
            pool.setdefault(profile.style, []).append(profile.font_id)
    return pool


def draw_printed_font(rng: np.random.Generator, kind: CheckSizeKind, role: PrintedRole, holdout: bool) -> str:
    """Draw a style by its weight (among styles present in the pool), then a font uniformly within it.

    Falls back to the regular pool if the held-out pool is empty for this kind and role.
    """
    pool = printed_font_pool(kind, role, holdout) or printed_font_pool(kind, role, holdout=False)
    styles = list(pool)
    weights = np.array([ROLE_STYLE_WEIGHTS_BY_KIND[kind][role][style] for style in styles])
    style = styles[int(rng.choice(len(styles), p=weights / weights.sum()))]
    return pool[style][int(rng.integers(len(pool[style])))]


def is_holdout_template_index(template_index: int) -> bool:
    """True for the template class that may (and mostly does) use held-out fonts."""
    return template_index % HOLDOUT_TEMPLATE_MODULUS == HOLDOUT_TEMPLATE_REMAINDER
