"""Builds the two test fixtures the Playwright smoke test needs.

Both fixtures are generated on demand rather than checked in as binary blobs, so the
repo stays free of test-data cruft and the fixtures are self-documenting (read this
file to know exactly what's in them).

- `oriented.jpg`: a landscape JPEG carrying an EXIF Orientation=6 tag. Orientation 6
  means "the stored pixels are rotated 90 degrees CCW from how they should display" —
  a viewer that honors EXIF rotates 90 degrees CW to show it correctly, which swaps
  width and height. This exercises `createImageBitmap(..., {imageOrientation:
  "from-image"})` in js/image_decode.js: if orientation handling is wrong, the
  reported "Photo received" dimensions come out unswapped.
- `fake.heic`: not a real decodable HEIC (no actual HEIF box structure beyond the
  header), just the exact magic bytes js/heic_detect.js looks for: bytes 4-8 spell
  "ftyp", bytes 8-12 carry the "heic" brand code. That's all the detector reads, and
  it's all this test needs — the app must never attempt to decode a HEIC file.
"""

from pathlib import Path

from PIL import Image

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# EXIF tag ID 274 is Orientation (see PIL.ExifTags.TAGS).
EXIF_ORIENTATION_TAG_ID = 274
EXIF_ORIENTATION_ROTATE_90_CW_ON_DISPLAY = 6

STORED_WIDTH_PX = 400
STORED_HEIGHT_PX = 300


def write_oriented_jpeg_fixture() -> Path:
    """Writes a 400x300 JPEG tagged EXIF Orientation=6, so a browser that honors it displays a 300x400 image."""
    image = Image.new("RGB", (STORED_WIDTH_PX, STORED_HEIGHT_PX), color=(246, 242, 234))
    # A corner marker, purely so a human eye-checking the fixture (or a future pixel-level
    # test) can tell which edge is "up" after rotation. Not read by the current test.
    for x in range(40):
        for y in range(40):
            image.putpixel((x, y), (63, 111, 91))

    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG_ID] = EXIF_ORIENTATION_ROTATE_90_CW_ON_DISPLAY

    output_path = FIXTURES_DIR / "oriented.jpg"
    image.save(output_path, "jpeg", exif=exif)
    return output_path


def write_fake_heic_fixture() -> Path:
    """Writes a minimal file with valid HEIC ftyp/brand magic bytes and nothing else."""
    box_size = (0).to_bytes(4, "big")  # unused by the detector; any 4 bytes will do
    ftyp_tag = b"ftyp"
    brand_code = b"heic"
    padding = b"\x00" * 8  # pad past the 12-byte magic-bytes read window in heic_detect.js

    output_path = FIXTURES_DIR / "fake.heic"
    output_path.write_bytes(box_size + ftyp_tag + brand_code + padding)
    return output_path


def main() -> None:
    FIXTURES_DIR.mkdir(exist_ok=True)
    oriented_path = write_oriented_jpeg_fixture()
    heic_path = write_fake_heic_fixture()
    print(f"wrote {oriented_path} ({oriented_path.stat().st_size} bytes)")
    print(f"wrote {heic_path} ({heic_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
