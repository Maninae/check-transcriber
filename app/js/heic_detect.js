/**
 * Detects HEIC/HEIF files so the app can show a hint instead of failing silently.
 *
 * No browser in scope (current Chrome or Edge on Windows) can decode HEIC via
 * `createImageBitmap`, so a HEIC photo must be caught before it reaches the decode
 * step. Detection is two-layered per the spec: file extension (fast, catches the
 * common case) and container magic bytes (catches a HEIC file that was renamed or
 * arrived with no extension, e.g. from some clipboard sources).
 */

// A HEIC/HEIF container is an ISO base media file: the first 4 bytes are a box
// size, bytes 4-8 spell "ftyp", and bytes 8-12 are a 4-character brand code that
// identifies the specific format. These are the brand codes real iPhones and iOS
// write for photo/HEIF content (image, not video/burst) as of iOS 18.
const HEIC_FTYP_BRANDS = ["heic", "heix", "heim", "heis", "mif1", "msf1", "heif"];

const FTYP_BOX_OFFSET = 4;
const FTYP_BRAND_OFFSET = 8;
const MAGIC_BYTES_READ_LENGTH = 12;

/**
 * True if the filename carries a HEIC/HEIF extension. Case-insensitive.
 */
export function hasHeicExtension(fileName) {
  return /\.(heic|heif)$/i.test(fileName);
}

/**
 * True if the file's container magic bytes identify it as HEIC/HEIF, independent
 * of its filename. Reads only the first 12 bytes, so this is cheap even on a
 * multi-megabyte photo.
 */
export async function hasHeicMagicBytes(file) {
  const header = await file.slice(0, MAGIC_BYTES_READ_LENGTH).arrayBuffer();
  if (header.byteLength < MAGIC_BYTES_READ_LENGTH) {
    return false;
  }
  const bytes = new Uint8Array(header);
  const ftypTag = String.fromCharCode(...bytes.slice(FTYP_BOX_OFFSET, FTYP_BOX_OFFSET + 4));
  if (ftypTag !== "ftyp") {
    return false;
  }
  const brand = String.fromCharCode(...bytes.slice(FTYP_BRAND_OFFSET, FTYP_BRAND_OFFSET + 4));
  return HEIC_FTYP_BRANDS.includes(brand);
}

/**
 * Combines both checks: true if either the extension or the magic bytes say HEIC.
 * OR (not AND) on purpose — a mislabeled file (say, a ".jpg" that's actually a HEIC
 * export) is still a HEIC file the browser can't decode, and should still get the hint.
 */
export async function isLikelyHeicImage(file) {
  if (hasHeicExtension(file.name)) {
    return true;
  }
  return hasHeicMagicBytes(file);
}
