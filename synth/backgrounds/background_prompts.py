"""Prompt vocabulary for generating household-surface background photos with FLUX.

The compositor pastes rendered checks onto these, so every background must be an EMPTY
surface shot from above, realistic like a snapshot, with nothing on it and no text anywhere.
The style sentence is frozen across the whole set; only surface, lighting and framing vary.

Prompt rules learned the hard way (Sep 25 2026), because FLUX paints every noun it reads:
- Never name a device ("phone", "camera"), even in a negative: it paints the device.
- Never name a light source ("lamp", "window", "flash"): it paints the lamp. Describe only
  what the light does to the surface (warmer, brighter on one side, a hotspot).
- Keep negatives to text and watermarks; "no paper, no hands" primes paper and hands.
"""

import random
from dataclasses import dataclass

STYLE_SENTENCE = (
    "Realistic snapshot photograph looking straight down at an empty household surface, "
    "the surface fills the entire frame edge to edge, natural fine texture, slightly imperfect framing"
)

NEGATIVES = "Nothing else in the frame, just the surface. No text, no words, no watermark."

# (slug, description, weight). Bedding carries about half the weight: that is what the
# property managers actually photograph checks on.
SURFACES: list[tuple[str, str, float]] = [
    ("white-bedsheet", "a plain white cotton bedsheet with soft wrinkles", 4.0),
    ("white-bedsheet-creased", "a white bedsheet with deep creases and fold lines", 2.5),
    ("ivory-bedsheet", "an ivory sateen bedsheet with a subtle sheen and loose ripples", 1.5),
    ("grey-bedsheet", "a light grey jersey bedsheet with gentle folds", 2.0),
    ("blue-bedsheet", "a pale blue percale bedsheet, slightly rumpled", 2.0),
    ("sage-bedsheet", "a sage green washed-linen bedsheet, crinkled", 1.0),
    ("navy-bedsheet", "a dark navy blue cotton bedsheet with soft folds", 1.0),
    ("striped-bedsheet", "a thin-striped cotton bedsheet with narrow blue stripes", 1.5),
    ("floral-bedsheet", "a faded floral-print bedsheet", 1.0),
    ("beige-duvet", "a beige quilted duvet cover with stitched squares", 1.0),
    ("white-comforter", "a puffy white down comforter with soft lumps", 1.0),
    ("plaid-blanket", "a plaid flannel blanket", 1.0),
    ("fleece-blanket", "a grey microfleece blanket with soft pile", 1.0),
    ("knit-throw", "a chunky knit throw blanket in cream", 0.5),
    ("wood-dining-table", "a warm oak dining table with visible wood grain", 1.5),
    ("dark-wood-table", "a dark walnut table top with a few scratches", 1.0),
    ("white-laminate-table", "a plain white laminate table top with a faint sheen", 1.5),
    ("birch-table", "a light birch wood table top", 1.0),
    ("desk-laminate", "a grey laminate office desk surface", 1.0),
    ("granite-countertop", "a speckled black and grey granite kitchen countertop", 1.0),
    ("marble-countertop", "a white marble countertop with grey veins", 1.0),
    ("butcher-block", "a butcher-block wooden kitchen counter", 0.7),
    ("laminate-countertop", "a beige laminate kitchen counter with a slight speckle pattern", 1.0),
    ("tile-counter", "a small square ceramic tile countertop with grout lines", 0.5),
    ("couch-cushion", "a grey woven fabric couch cushion, close up", 1.0),
    ("leather-couch", "a brown leather sofa seat, close up", 0.5),
    ("linen-couch", "a beige linen sofa cushion with visible weave", 1.0),
    ("carpet", "a low-pile beige carpet", 0.8),
    ("area-rug", "a patterned woven area rug in muted red and blue", 0.5),
    ("hardwood-floor", "a hardwood floor with planks running diagonally", 0.8),
    ("checkered-tablecloth", "a checkered red and white cotton tablecloth", 0.5),
    ("linen-tablecloth", "a natural linen tablecloth with soft wrinkles", 1.0),
    ("cardboard", "a flat sheet of brown cardboard", 0.4),
    ("desk-mat", "a black felt desk mat", 0.4),
]

# Light is described by what it does to the surface, never by its source (see module docstring).
LIGHTING: list[str] = [
    "soft diffuse daylight, slightly brighter along the left side",
    "warm amber indoor light, brighter toward the top and dimmer toward the bottom",
    "flat even neutral indoor lighting",
    "a bright hotspot in the middle fading to darker corners",
    "dim warm indoor lighting with one corner falling into soft shadow",
    "bright neutral light with a gentle falloff toward one edge",
    "cool bluish light, slightly underexposed",
    "mixed light, warm on one side of the frame and cool on the other",
]

FRAMING: list[str] = [
    "seen perfectly from overhead",
    "seen from above with a slight tilt so the far edge looks a little smaller",
    "seen from above at a small angle with mild perspective",
]


@dataclass(frozen=True)
class BackgroundPrompt:
    """One background to generate: the full prompt text plus the slug used in its filename."""

    slug: str
    prompt: str


def build_background_prompt(rng: random.Random) -> BackgroundPrompt:
    """Draw one weighted surface, a lighting and a framing, and assemble the frozen-style prompt."""
    slug, surface, _ = rng.choices(SURFACES, weights=[weight for _, _, weight in SURFACES])[0]
    lighting = rng.choice(LIGHTING)
    framing = rng.choice(FRAMING)
    prompt = f"{STYLE_SENTENCE}. {surface}, {lighting}, {framing}. {NEGATIVES}"
    return BackgroundPrompt(slug=slug, prompt=prompt)
