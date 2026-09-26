"""Search terms for real photos (Openverse, Commons) and the title rules that skip obvious misfits.

Each term carries the surface category its hits are filed under. Titles are screened before any
download: texture-site mirrors (already fetched directly), non-colour PBR maps, museum object
photography (antique quilts on backdrops) and icons.
"""

import re

from synth.backgrounds.web_candidate import SurfaceCategory

PHOTO_SEARCH_TERMS: tuple[tuple[str, SurfaceCategory], ...] = (
    ("bedsheet", SurfaceCategory.BEDDING),
    ("bed sheet top view", SurfaceCategory.BEDDING),
    ("bed sheets", SurfaceCategory.BEDDING),
    ("duvet cover", SurfaceCategory.BEDDING),
    ("blanket texture", SurfaceCategory.BEDDING),
    ("blanket", SurfaceCategory.BEDDING),
    ("quilt", SurfaceCategory.BEDDING),
    ("linen tablecloth", SurfaceCategory.TABLE_LINEN),
    ("tablecloth", SurfaceCategory.TABLE_LINEN),
    ("linen texture", SurfaceCategory.FABRIC),
    ("fabric texture", SurfaceCategory.FABRIC),
    ("cotton fabric", SurfaceCategory.FABRIC),
    ("couch cushion fabric", SurfaceCategory.FABRIC),
    ("wooden table top", SurfaceCategory.WOOD_TABLE),
    ("wood texture", SurfaceCategory.WOOD_TABLE),
    ("desk surface", SurfaceCategory.WOOD_TABLE),
    ("kitchen counter top view", SurfaceCategory.STONE_COUNTER),
    ("granite countertop", SurfaceCategory.STONE_COUNTER),
    ("marble countertop", SurfaceCategory.STONE_COUNTER),
    ("marble texture", SurfaceCategory.STONE_COUNTER),
    ("carpet texture", SurfaceCategory.RUG_CARPET),
    ("rug texture", SurfaceCategory.RUG_CARPET),
    ("cardboard texture", SurfaceCategory.CARDBOARD),
)

EXCLUDED_TITLE = re.compile(
    r"poly ?haven|ambientcg|unsplash|\b\d{1,2}K\b|\b(nor|nor gl|nor dx|disp|rough|arm|ao|metal|anisotropy|spec|bump|normal|displacement|"
    r"roughness|clay|reference|primary flat|icon|logo|map|diagram|radiograph|x-ray)\b|"
    r"preview|collectie|yale center|batik|\bpage\b|diar(y|ies)|letter|journal|crater|cultivation|picker|harvest|"
    r"church|bridge|village|\bshop\b|clevelandart|NMAH|RP-P-|\bBK-\d|firewood|bowl|wall hanging|vestments|"
    r"\bMET\b|\bNGA\b|museum|accession|ARTIC|SAAM|\bLACMA\b|collection|painting|engraving|lithograph|drawing|poster|"
    r"(?-i:\bby [A-Z])",
    re.IGNORECASE,
)
# Photo titles naming a subject other than the surface (people, pets, devices, food, rooms).
SUBJECT_WORDS_IN_TITLE = re.compile(
    r"\b(people|person|woman|women|man|men|girl|boy|child|children|kid|baby|babies|family|couple|portrait|hand|hands|"
    r"feet|foot|dog|dogs|puppy|pug|cat|cats|kitten|animal|bird|laptop|macbook|iphone|phone|mobile|computer|camera|"
    r"book|books|coffee|tea|cup|mug|food|breakfast|dinner|lunch|cake|fruit|bread|pizza|wine|drink|flower|flowers|"
    r"plant|christmas|gift|present|bedroom|room|hotel|house|home|interior|kitchen|living|cabin|logcabin|landscape|"
    r"nature|outdoor|beach|city|street|car|hat|shoe|shoes|fashion|dress|clothes|clothing|students?|ward|hospital|"
    r"soldiers?|army|navy|war|sleeping|sleep|bear|toy|toys)\b",
    re.IGNORECASE,
)


def is_excluded_title(title: str) -> bool:
    """True for texture-site mirrors, non-colour maps, museum objects, artworks and photos titled after another subject."""
    return bool(EXCLUDED_TITLE.search(title) or SUBJECT_WORDS_IN_TITLE.search(title))
