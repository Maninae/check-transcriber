"""Invented payees (land trusts, housing co-ops, property LLCs) and invented banks.

Both lists are partitioned across train / val / eval like fonts (contract C4, `splits.py`), so
OCR eval on the payee and bank_name fields reads names train never saw instead of measuring
recall of a closed list. Every name is made up; mailing-address ZIPs end in 999 on purpose.

- `PAYEES`: canonical name -> spellings a tenant might write, plus the window-envelope address
  printed under the payee on business stock.
- `BANK_NAMES`: printed in the bank block of every check.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FakePayee:
    """One invented payee: its canonical name, how people write it, and its mailing address."""

    canonical_name: str
    spellings: tuple[str, ...]
    mailing_address_lines: tuple[str, str]


PAYEES: tuple[FakePayee, ...] = (
    FakePayee("Quailbrook Community Land Trust", ("Quailbrook Community Land Trust", "Quailbrook CLT", "Quailbrook Land Trust"),
              ("PO Box 4418", "Quailbrook, OR 97999")),
    FakePayee("Fennimore Street Housing Cooperative", ("Fennimore Street Housing Cooperative", "Fennimore St. Housing Co-op", "Fennimore Co-op"),
              ("212 Fennimore St, Office 2", "Larchmont Falls, WI 53999")),
    FakePayee("Driftwood Commons Land Trust", ("Driftwood Commons Land Trust", "Driftwood Commons", "Driftwood Commons LT"),
              ("88 Driftwood Commons Way", "Seacliff Harbor, ME 04999")),
    FakePayee("Marrowstone Co-op Homes", ("Marrowstone Co-op Homes", "Marrowstone Co-op", "Marrowstone Coop Homes"),
              ("1400 Marrowstone Loop", "Tidewater Bend, WA 98999")),
    FakePayee("Saltgrass Community Land Trust", ("Saltgrass Community Land Trust", "Saltgrass CLT"),
              ("PO Box 2207", "Saltgrass Flats, NM 87999")),
    FakePayee("Hollis Yard Housing Co-op", ("Hollis Yard Housing Co-op", "Hollis Yard Co-op", "Hollis Yard Housing"),
              ("9 Hollis Yard, Suite B", "Brickmill, PA 19999")),
    FakePayee("Thistledown Mutual Housing Association", ("Thistledown Mutual Housing Association", "Thistledown MHA", "Thistledown Mutual Housing"),
              ("PO Box 761", "Owlhead Crossing, VT 05999")),
    FakePayee("Larkspur Terrace Properties LLC", ("Larkspur Terrace Properties LLC", "Larkspur Terrace Properties", "Larkspur Terrace"),
              ("530 Larkspur Terrace", "Juniper Mesa, AZ 85999")),
    FakePayee("Kettle Pond Housing Cooperative", ("Kettle Pond Housing Cooperative", "Kettle Pond Co-op", "Kettle Pond Housing"),
              ("17 Kettle Pond Rd", "Birchfold, MN 55999")),
    FakePayee("Oxbow Bend Community Land Trust", ("Oxbow Bend Community Land Trust", "Oxbow Bend CLT", "Oxbow Bend Land Trust"),
              ("PO Box 3390", "Oxbow Bend, MO 63999")),
    FakePayee("Cinderhill Apartments LLC", ("Cinderhill Apartments LLC", "Cinderhill Apartments", "Cinderhill Apts"),
              ("2210 Cinderhill Ave", "Ember Falls, ID 83999")),
    FakePayee("Wrenfield Cooperative Homes", ("Wrenfield Cooperative Homes", "Wrenfield Co-op Homes", "Wrenfield Co-op"),
              ("44 Wrenfield Ct", "Hazelmoor, NC 27999")),
    FakePayee("Blue Heron Commons Land Trust", ("Blue Heron Commons Land Trust", "Blue Heron Commons", "Blue Heron CLT"),
              ("PO Box 912", "Marshgate, SC 29999")),
    FakePayee("Tamarack Row Holdings LLC", ("Tamarack Row Holdings LLC", "Tamarack Row Holdings", "Tamarack Row"),
              ("75 Tamarack Row, Suite 3", "Pinecleft, MT 59999")),
    FakePayee("Millrace Street Housing Co-op", ("Millrace Street Housing Co-op", "Millrace St. Co-op", "Millrace Housing Co-op"),
              ("301 Millrace St", "Wheelwright, OH 44999")),
    FakePayee("Sorrel Creek Community Land Trust", ("Sorrel Creek Community Land Trust", "Sorrel Creek CLT", "Sorrel Creek Land Trust"),
              ("PO Box 1185", "Sorrel Creek, KY 40999")),
    FakePayee("Beacon Knoll Residential LLC", ("Beacon Knoll Residential LLC", "Beacon Knoll Residential", "Beacon Knoll"),
              ("890 Beacon Knoll Dr", "Lanternport, RI 02999")),
    FakePayee("Aldergrove Mutual Housing", ("Aldergrove Mutual Housing", "Aldergrove Housing", "Aldergrove MH"),
              ("12 Aldergrove Ln", "Fernhollow, OR 97999")),
    FakePayee("Cobblestone Yard Cooperative", ("Cobblestone Yard Cooperative", "Cobblestone Yard Co-op", "Cobblestone Co-op"),
              ("6 Cobblestone Yard", "Old Quarry, MA 01999")),
    FakePayee("Prairie Lantern Land Trust", ("Prairie Lantern Land Trust", "Prairie Lantern LT", "Prairie Lantern"),
              ("PO Box 5520", "Windrow, KS 66999")),
    FakePayee("Halyard Point Properties LLC", ("Halyard Point Properties LLC", "Halyard Point Properties", "Halyard Point"),
              ("1 Halyard Point Wharf", "Keelhaven, CT 06999")),
    FakePayee("Greenwillow Housing Cooperative", ("Greenwillow Housing Cooperative", "Greenwillow Co-op", "Greenwillow Housing"),
              ("420 Greenwillow Pkwy", "Riverlight, IA 50999")),
    FakePayee("Stonecrop Community Land Trust", ("Stonecrop Community Land Trust", "Stonecrop CLT", "Stonecrop Land Trust"),
              ("PO Box 648", "Graniteview, NH 03999")),
    FakePayee("Maple Hollow Rentals LLC", ("Maple Hollow Rentals LLC", "Maple Hollow Rentals", "Maple Hollow"),
              ("3345 Maple Hollow Rd", "Sugarbush, MI 49999")),
    FakePayee("Rookery Lane Co-op Homes", ("Rookery Lane Co-op Homes", "Rookery Lane Co-op", "Rookery Lane Homes"),
              ("58 Rookery Ln", "Crowsnest, WY 82999")),
    FakePayee("Eastwater Mutual Housing Association", ("Eastwater Mutual Housing Association", "Eastwater MHA", "Eastwater Mutual Housing"),
              ("PO Box 2044", "Eastwater, DE 19999")),
    FakePayee("Sunbarrow Court Apartments LLC", ("Sunbarrow Court Apartments LLC", "Sunbarrow Court Apartments", "Sunbarrow Ct Apts"),
              ("150 Sunbarrow Ct", "Dunmeadow, NV 89999")),
    FakePayee("Foxglove Commons Housing Co-op", ("Foxglove Commons Housing Co-op", "Foxglove Commons Co-op", "Foxglove Co-op"),
              ("27 Foxglove Commons", "Briarwick, VA 24999")),
    FakePayee("Lindenmoor Community Land Trust", ("Lindenmoor Community Land Trust", "Lindenmoor CLT", "Lindenmoor Land Trust"),
              ("PO Box 7301", "Lindenmoor, NE 68999")),
    FakePayee("Ironwood Flats Property Group LLC", ("Ironwood Flats Property Group LLC", "Ironwood Flats Property Group", "Ironwood Flats"),
              ("9900 Ironwood Flats Blvd", "Anvil Springs, TX 79999")),
)

PAYEE_BY_NAME: dict[str, FakePayee] = {payee.canonical_name: payee for payee in PAYEES}
PAYEE_NAMES: tuple[str, ...] = tuple(PAYEE_BY_NAME)

BANK_NAMES: tuple[str, ...] = (
    "First Meridian Bank",
    "Pacific Tidewater Credit Union",
    "Cedar Valley Savings Bank",
    "Golden Bluff Bank, N.A.",
    "Harborline Federal Credit Union",
    "Summit Ridge Bank",
    "Redwood Crossing Bank",
    "Copper Canyon Savings",
    "Bayshore Mutual Bank",
    "Northgate Trust Bank",
    "Lanternfield State Bank",
    "Silver Birch Community Bank",
    "Keystone Prairie Bank",
    "Osprey Bay Credit Union",
    "Granite Hollow Savings & Loan",
    "Ember Plains National Bank",
    "Whitewater Junction Bank",
    "Clearbrook Federal Savings",
    "Tallgrass Farmers Bank",
    "Seven Rivers Credit Union",
    "Highmoor Trust Company",
    "Sandpiper Coast Bank",
    "Juniper Heights Bank, N.A.",
    "Lodestone Mutual Savings",
    "Brightwater Valley Credit Union",
    "Old Foundry Bank & Trust",
    "Cascade Hearth Bank",
    "Palisade Point Savings Bank",
    "Windmill Ridge Community Bank",
    "Starling Harbor Federal Credit Union",
)
