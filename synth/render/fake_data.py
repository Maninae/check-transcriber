"""Invent the values written on a check. Everything here is fake by construction.

- Names and addresses come from Faker (made-up combinations).
- Payees and banks are invented (`fake_payees_and_banks.py`); callers pass the split's own
  payee and bank pools (contract C4), defaulting to every name.
- Money orders follow issuer rules: the issuer prints number, date and amount (capped at $1,000);
  the purchaser writes payee, memo and their own name and address.
- Routing numbers are 9 digits that deliberately FAIL the ABA checksum, so no generated
  check can ever carry a real bank's routing number.
"""

import datetime
import functools

import numpy as np
from faker import Faker

from synth.render.amount_words import spell_whole_dollars
from synth.render.check_fields import HANDWRITABLE_FIELDS, CheckContent
from synth.render.check_layout import LayoutFamily
from synth.render.check_templates import TemplateDesign
from synth.render.fake_payees_and_banks import BANK_NAMES, PAYEE_BY_NAME, PAYEE_NAMES
from synth.render.fonts import MICR_ON_US_SYMBOL, MICR_TRANSIT_SYMBOL

MEMO_TEMPLATES = [
    "", "", "", "{month} rent", "Unit {unit}", "Rent - {month}", "Apt {unit} rent",
    "#{unit} {month}", "rent", "{month} rent Unit {unit}", "{month} {year} rent", "Unit {unit} - {month}",
    "{month} rent #{unit}", "rent {month_number}/{short_year}", "{month}. rent", "Rent {month} {year} - Apt {unit}",
    "{unit}", "Unit {unit} {month} rent", "rent + parking", "{full_month}", "{month} {year}", "Apt. {unit}",
]
MONEY_ORDER_MAX_DOLLARS = 1000
PERSONAL_HANDWRITTEN_PROBABILITY = 0.9   # the rest are typed by bill-pay or check-writing software
BUSINESS_HANDWRITTEN_PROBABILITY = 0.15  # most business checks are printed by accounting software
MONTH_ABBREVIATIONS = ["Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]

RENT_MIN_DOLLARS = 450
RENT_MAX_DOLLARS = 3200
DATE_RANGE_START = datetime.date(2025, 1, 1)
DATE_RANGE_DAYS = 900
HANDWRITTEN_INK_RGB = [(20, 30, 90), (25, 25, 30), (10, 40, 120), (40, 40, 60)]
PRINTED_INK_RGB = (22, 22, 26)


def aba_checksum_is_valid(routing_number: str) -> bool:
    """ABA routing checksum: 3(d1+d4+d7) + 7(d2+d5+d8) + (d3+d6+d9) must be divisible by 10."""
    digits = [int(character) for character in routing_number]
    weighted_sum = 3 * (digits[0] + digits[3] + digits[6]) + 7 * (digits[1] + digits[4] + digits[7]) + digits[2] + digits[5] + digits[8]
    return weighted_sum % 10 == 0


def fake_routing_number(rng: np.random.Generator) -> str:
    """Nine digits with a plausible Fed-district prefix and a deliberately invalid checksum."""
    prefix = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 26, 31, 32]))
    digits = f"{prefix:02d}" + "".join(str(int(d)) for d in rng.integers(0, 10, 7))
    if aba_checksum_is_valid(digits):
        # Bumping the last digit changes the weighted sum by 1, which always breaks divisibility.
        digits = digits[:8] + str((int(digits[8]) + 1) % 10)
    return digits


# Families whose stock preprints "$" beside the amount box; their fill must not add a second one.
FAMILIES_WITH_PREPRINTED_DOLLAR_SIGN = frozenset({
    LayoutFamily.PERSONAL_CLASSIC,
    LayoutFamily.PERSONAL_DATE_BOX,
    LayoutFamily.PERSONAL_NAME_ONLY,
})


def format_amount_numeric(amount_cents: int, rng: np.random.Generator, handwritten: bool, money_order: bool = False,
                          preprinted_dollar_sign: bool = False) -> str:
    """Courtesy-box amount in one of the styles people, check printers and money-order machines use.

    With `preprinted_dollar_sign` the stock already prints "$", so the fill never starts with another.
    """
    dollars, cents = divmod(amount_cents, 100)
    if money_order:
        styles = [f"${dollars:,}.{cents:02d}", f"**{dollars:,}.{cents:02d}**", f"$***{dollars:,}.{cents:02d}"]
    elif handwritten:
        styles = [f"{dollars:,}.{cents:02d}", f"{dollars:,}.{cents:02d}", f"{dollars}.{cents:02d}", f"{dollars:,} {cents:02d}/100",
                  f"{dollars} {cents:02d}/100", f"${dollars:,}.{cents:02d}"]
        if cents == 0:  # whole-dollar habits: "1,250.-", "1250 =", "1,250 xx/100", "1,250.xx"
            styles += [f"{dollars:,}.-", f"{dollars} =", f"{dollars:,} xx/100", f"{dollars:,}.xx", f"{dollars:,}"]
    else:
        styles = [f"{dollars:,}.{cents:02d}", f"{dollars}.{cents:02d}", f"**{dollars:,}.{cents:02d}", f"***{dollars:,}.{cents:02d}",
                  f"$*****{dollars:,}.{cents:02d}"]
    if preprinted_dollar_sign:
        styles = [style.removeprefix("$") for style in styles]
    return styles[int(rng.integers(len(styles)))]


def format_amount_words(amount_cents: int, rng: np.random.Generator, handwritten: bool, money_order: bool = False) -> str:
    """Legal-line amount, e.g. 'One thousand two hundred fifty and 00/100'."""
    dollars, cents = divmod(amount_cents, 100)
    words = spell_whole_dollars(dollars)
    if money_order:
        return [f"{words.upper()} DOLLARS AND {cents:02d} CENTS", f"***{words.upper()} AND {cents:02d}/100***",
                f"{words.title()} Dollars {cents:02d} Cents"][int(rng.integers(3))]
    if handwritten:  # sentence case, no asterisks; the renderer adds the long hand-drawn dash, unlabelled
        words = words.capitalize() if rng.random() < 0.85 else words
        cents_parts = ["and {c:02d}/100", "and {c:02d}/100", "& {c:02d}/100"]
        if cents == 0:
            cents_parts += ["and no/100", "& xx/100", "and xx/100"]
        return f"{words} {cents_parts[int(rng.integers(len(cents_parts)))].format(c=cents)}"
    words = words.title() if rng.random() < 0.5 else words.upper()
    return f"***{words} and {cents:02d}/100***"


def format_check_date(date_value: datetime.date, rng: np.random.Generator, handwritten: bool) -> str:
    """Date in one of the formats seen on US checks."""
    month, day, year = date_value.month, date_value.day, date_value.year
    short_month, full_month = MONTH_ABBREVIATIONS[month - 1], MONTH_NAMES[month - 1]
    styles = [f"{month}/{day}/{year}", f"{month:02d}/{day:02d}/{year}", f"{month}-{day}-{year % 100:02d}",
              f"{short_month} {day}, {year}", f"{full_month} {day}, {year}", f"{month}/{day}/{year % 100:02d}"]
    if handwritten:
        styles += [f"{month:02d}-{day:02d}-{year}", f"{short_month} {ordinal(day)} {year}", f"{short_month}. {day}, {year}",
                   f"{full_month} {ordinal(day)}, {year}", f"{month}.{day}.{year % 100:02d}", f"{month}/{day}/{year % 100:02d}"]
    else:
        styles += [date_value.isoformat(), f"{short_month[:3].upper()} {day:02d} {year}", f"{month:02d}/{day:02d}/{year % 100:02d}"]
    return styles[int(rng.integers(len(styles)))]


def ordinal(day: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 23 -> '23rd'."""
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def sample_rent_amount_cents(rng: np.random.Generator) -> int:
    """Rent in a realistic range; most are whole dollars, many are multiples of 25."""
    dollars = int(rng.integers(RENT_MIN_DOLLARS, RENT_MAX_DOLLARS + 1))
    roll = rng.random()
    if roll < 0.45:
        dollars = round(dollars / 25) * 25
    cents = int(rng.integers(1, 100)) if roll > 0.85 else 0
    return dollars * 100 + cents


def build_micr_lines(routing_number: str, account_number: str, check_number: str, micr_layout: str) -> tuple[str, str]:
    """Return (GnuMICR-typed line, human-readable line with Unicode E-13B symbols)."""
    transit, on_us = MICR_TRANSIT_SYMBOL, MICR_ON_US_SYMBOL
    if micr_layout == "business":
        font_text = f"{on_us}{check_number.zfill(6)}{on_us} {transit}{routing_number}{transit} {account_number}{on_us}"
    else:
        font_text = f"{transit}{routing_number}{transit} {account_number}{on_us} {check_number}"
    readable_text = font_text.replace(transit, "⑆").replace(on_us, "⑈")
    return font_text, readable_text


@functools.lru_cache(maxsize=1)
def shared_faker() -> Faker:
    """One Faker per process (construction is slow); callers reseed it per check."""
    return Faker("en_US")


def sample_check_content(template: TemplateDesign, rng: np.random.Generator, serial: str | None = None,
                         payee_names: tuple[str, ...] | list[str] = PAYEE_NAMES,
                         bank_names: tuple[str, ...] | list[str] = BANK_NAMES) -> CheckContent:
    """Invent every value for one check drawn on `template`.

    `payee_names` / `bank_names` are the split's pools (contract C4); the default is every name.
    """
    faker = shared_faker()
    faker.seed_instance(int(rng.integers(2**31)))
    is_business = template.micr_layout == "business"
    is_money_order = template.layout_family == LayoutFamily.MONEY_ORDER
    if is_money_order:
        payer_name = faker.name()  # written by hand by the purchaser, so natural case
    elif is_business and rng.random() < 0.7:
        payer_name = faker.company().upper() if rng.random() < 0.5 else faker.company()
    elif rng.random() < 0.25:
        payer_name = f"{faker.first_name()} & {faker.first_name()} {faker.last_name()}"
    else:
        payer_name = faker.name()
    if not is_money_order and rng.random() < 0.35:
        payer_name = payer_name.upper()
    payer_address_lines = [faker.street_address(), f"{faker.city()}, {faker.state_abbr(include_territories=False, include_freely_associated_states=False)} {faker.zipcode()}"]

    # One fill mode per check: a person writes every fill-in, or software types every one (never mixed,
    # except money orders, where the issuer prints date and amount and the purchaser writes the rest).
    handwritten_probability = BUSINESS_HANDWRITTEN_PROBABILITY if is_business else PERSONAL_HANDWRITTEN_PROBABILITY
    written_by_hand = rng.random() < handwritten_probability
    handwritten_fields = [field_name.value for field_name in HANDWRITABLE_FIELDS] if written_by_hand else []
    if is_money_order:  # the issuer printed date and amount; the purchaser writes payee and memo
        handwritten_fields = ["payee", "memo"]

    amount_cents = sample_rent_amount_cents(rng)
    if is_money_order and amount_cents > MONEY_ORDER_MAX_DOLLARS * 100:  # issuers cap one money order at $1,000
        amount_cents = int(rng.integers(RENT_MIN_DOLLARS, MONEY_ORDER_MAX_DOLLARS + 1)) * 100
    date_value = DATE_RANGE_START + datetime.timedelta(days=int(rng.integers(DATE_RANGE_DAYS)))
    payee_canonical = payee_names[int(rng.integers(len(payee_names)))]
    spellings = PAYEE_BY_NAME[payee_canonical].spellings
    payee_text = spellings[int(rng.integers(len(spellings)))]
    unit = f"{int(rng.integers(1, 30))}{'ABCD'[int(rng.integers(4))] if rng.random() < 0.6 else ''}"
    memo_text = MEMO_TEMPLATES[int(rng.integers(len(MEMO_TEMPLATES)))].format(
        month=MONTH_ABBREVIATIONS[date_value.month - 1], unit=unit, year=date_value.year, month_number=date_value.month,
        short_year=f"{date_value.year % 100:02d}", full_month=MONTH_NAMES[date_value.month - 1])

    if is_money_order:
        check_number = "".join(str(int(d)) for d in rng.integers(0, 10, 11)).lstrip("0") or "1"
    else:
        check_number = str(int(rng.integers(1001, 99999)) if is_business else int(rng.integers(101, 9999)))
    routing_number = fake_routing_number(rng)
    account_number = "".join(str(int(d)) for d in rng.integers(0, 10, int(rng.integers(8, 13))))
    micr_font_text, micr_readable_text = build_micr_lines(routing_number, account_number, check_number, template.micr_layout)

    return CheckContent(
        payer_name=payer_name,
        payer_address_lines=payer_address_lines,
        check_number=check_number,
        date_text=format_check_date(date_value, rng, "date" in handwritten_fields),
        date_iso=date_value.isoformat(),
        payee_text=payee_text,
        payee_canonical=payee_canonical,
        amount_cents=amount_cents,
        amount_numeric_text=format_amount_numeric(amount_cents, rng, "amount_numeric" in handwritten_fields, is_money_order,
                                                  template.layout_family in FAMILIES_WITH_PREPRINTED_DOLLAR_SIGN),
        amount_words_text=format_amount_words(amount_cents, rng, "amount_words" in handwritten_fields, is_money_order),
        bank_name=bank_names[int(rng.integers(len(bank_names)))],
        bank_city_line=f"{faker.city()}, {faker.state_abbr(include_territories=False, include_freely_associated_states=False)}",
        memo_text=memo_text,
        signature_text=payer_name.title() if not is_business or is_money_order else faker.name(),
        routing_number=routing_number,
        account_number=account_number,
        micr_font_text=micr_font_text,
        micr_readable_text=micr_readable_text,
        handwritten_fields=handwritten_fields,
        ink_rgb=HANDWRITTEN_INK_RGB[int(rng.integers(len(HANDWRITTEN_INK_RGB)))],
        serial=serial,
    )
