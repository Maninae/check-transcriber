"""Spell a dollar amount the way it appears on a check's legal line."""

ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def spell_below_thousand(number: int) -> str:
    """English words for 0 < number < 1000."""
    hundreds, remainder = divmod(number, 100)
    parts = []
    if hundreds:
        parts.append(f"{ONES[hundreds]} hundred")
    if remainder >= 20:
        tens_word = TENS[remainder // 10]
        parts.append(f"{tens_word}-{ONES[remainder % 10]}" if remainder % 10 else tens_word)
    elif remainder:
        parts.append(ONES[remainder])
    return " ".join(parts)


def spell_whole_dollars(dollars: int) -> str:
    """English words for 0 <= dollars < 1,000,000 ("zero" for 0)."""
    if dollars == 0:
        return "zero"
    thousands, remainder = divmod(dollars, 1000)
    parts = []
    if thousands:
        parts.append(f"{spell_below_thousand(thousands)} thousand")
    if remainder:
        parts.append(spell_below_thousand(remainder))
    return " ".join(parts)
