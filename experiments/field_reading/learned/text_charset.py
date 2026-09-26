"""Character sets for the CTC recognizers and the text <-> label-index mapping.

- Index 0 is always the CTC blank; character i of the charset string maps to label i + 1.
- GENERAL_CHARSET covers every ground-truth character of the target fields and bank_name in
  train, val and eval (verified identical across splits). MICR glyphs are never read.
- AMOUNT_CHARSET is the courtesy-box vocabulary (`$***1,250.00`, `2,475 00/100`, `725.-`,
  `2979 =`, `694 xx/100`) plus `X o n` so hand-written variants of "no/100" stay representable.
- Characters outside a charset are dropped on encode (logged by the caller if it cares).
"""

from dataclasses import dataclass, field

CTC_BLANK_INDEX = 0

GENERAL_CHARSET = (" #$&*+,-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                   "abcdefghijklmnopqrstuvwxyz")
AMOUNT_CHARSET = " $*,-./0123456789=xXon"


@dataclass(frozen=True)
class TextCharset:
    """An ordered character vocabulary for a CTC head (blank at index 0)."""

    name: str
    characters: str
    character_to_label: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(set(self.characters)) != len(self.characters):
            raise ValueError(f"charset {self.name} has duplicate characters")
        object.__setattr__(self, "character_to_label",
                           {character: index + 1 for index, character in enumerate(self.characters)})

    @property
    def num_classes(self) -> int:
        """Output classes of the CTC head: every character plus the blank."""
        return len(self.characters) + 1

    def encode(self, text: str) -> list[int]:
        """Label indices for `text`; characters outside the charset are skipped."""
        return [self.character_to_label[character] for character in text if character in self.character_to_label]

    def decode_labels(self, labels: list[int]) -> str:
        """Text for already-collapsed label indices (blanks are ignored)."""
        return "".join(self.characters[label - 1] for label in labels if label != CTC_BLANK_INDEX)

    def covers(self, text: str) -> bool:
        """True if every character of `text` is representable."""
        return all(character in self.character_to_label for character in text)


CHARSET_REGISTRY: dict[str, TextCharset] = {
    "general": TextCharset("general", GENERAL_CHARSET),
    "amount": TextCharset("amount", AMOUNT_CHARSET),
}
