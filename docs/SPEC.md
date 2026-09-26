# Check Transcriber: Spec (draft 1, Sep 25 2026)

A browser-only tool that turns a phone photo of several checks into upright, cropped, readable check images with the key fields pulled out, reviewed by a person, and copied into a spreadsheet. The person using it is referred to below as the operator.

## 1. Goal

Several checks arrive as one phone photo: a handful of checks laid on a plain surface, photographed from above, sent by email. Someone opens the email, reads each check off the photo by eye, and types who paid, how much, the date and the check number into a spreadsheet.

The tool replaces the reading-by-eye step. The operator pastes the photo into a web page, the page finds each check, straightens it, reads the printed fields, leaves the uncertain ones blank, and shows everything on one scrolling page for the operator's to confirm and copy. Target: a six-check batch goes from ten minutes of squinting to under two minutes of confirming.

Success looks like: it gets used for every batch without being asked, nothing ever has to be installed or updated, and no data ever leaves the laptop, so there is nothing to approve or share.

## 2. Constraints and non-goals

Constraints:
- Runs entirely in the browser. No backend, no server, no account, no login. Hosted as static files on GitHub Pages. The page must work with the network tab silent after first load.
- No installation. No terminal, no Homebrew, no downloaded app, no browser extension. A link is the whole distribution.
- An ordinary laptop, Chrome or Edge, current versions. Everything must work without WebGPU; WebGPU is an optional accelerator only.
- Input is a phone photo (JPEG, typically 3 to 12 megapixels), arriving as a Gmail attachment.
- The UI never says "AI" or "model." The page describes itself as a check scanner. Every automatic read is visibly provisional until the operator confirms it.
- Check images contain bank routing and account numbers. The tool never extracts, stores or displays the MICR line (the machine-printed numbers along the bottom edge) beyond what is needed to find the check number, and by default blurs that band in every crop it shows.
- Zero maintenance burden. No dependency that phones home, no service that expires, no build step required to keep it running. If nobody touches the repo for two years it still works.

Non-goals (v1):
- Reading handwriting reliably. Handwritten amount and date are best-effort; blank-with-crop is the designed fallback.
- Writing into Google Sheets. Output is clipboard text the operator pastes herself.
- Multi-page or multi-photo batches. One photo at a time. (A batch of several photos is a natural v2.)
- Anything to do with the deposit itself, the bank, or accounting.

## 3. First visit

- No permission prompts of any kind. Fetching the OpenCV and OCR files is a normal page load, so the browser never asks. Paste and drag-drop need no permission. Writing to the clipboard from a button click is allowed without a prompt in Chrome and Edge.
- On first load the page shows the drop zone immediately and a quiet progress line beneath it: "Getting ready for the first time (about 15 MB, once)." The page is usable the moment the files are cached; later visits load instantly from the browser cache. If the files fail to load (offline on first visit), the page says so plainly and offers Retry.
- One line under the drop zone, always visible: "Photos stay on this computer. Nothing is uploaded." This is the trust sentence. It is true because there is no server, and the operator can verify it in the browser's network tab if the operator ever wants to.
- No tour, no onboarding modal, no cookie banner (there are no cookies).

## 4. Typical flow

Three screens, in order, on one page that scrolls. The top of the page always shows which step the operator is on and the check count once known.

### 4.1 Getting the photo in

The page starts as one large drop zone. Three doors, all landing in the same place:
- Paste. In Gmail, click the attachment to open the preview, right-click the image, Copy image, switch to the tool's tab, Ctrl+V anywhere on the page. This is the fast path and the one to teach the operator's once.
- Drag. Drag the attachment thumbnail from Gmail (or a file from Explorer) onto the page.
- Browse. Click the drop zone to open a file picker.

Details:
- Accept JPEG, PNG, WebP. If the file is HEIC (an iPhone original that was downloaded rather than previewed), the browser cannot decode it; show a one-line hint: "This is an iPhone HEIC file. Open the attachment in Gmail and use Copy image instead." Detect by file extension and by magic bytes.
- Honor EXIF orientation so a photo taken in portrait is not processed sideways. Then strip EXIF from the in-memory copy (it can carry GPS coordinates).
- Downscale very large photos to a working size of about 2500 px on the long edge for detection, but keep the full-resolution image for the crops so the checks stay legible.
- A pasted image from Gmail's preview is already a rendered JPEG at reasonable size; this is the common case.
- Replacing the photo (paste again) asks "Start over with a new photo? The current batch will be cleared." if there is an unfinished batch.

### 4.2 Confirming the count

The photo is shown at fit-to-width with a numbered outline around each detected check and a header: "Found 6 checks." Numbering is reading order: top-to-bottom, then left-to-right within a row, so it matches how the operator would scan the bedsheet.

The operator can fix detection here, cheaply:
- Add a missed check: click "Add a check" then drag a rectangle on the photo. The tool re-fits the rectangle to the nearest strong edges so a rough drag becomes a tight box.
- Remove a false detection: click its outline, then the small X on the outline (or press Delete). Typical false positives: a deposit slip, an envelope, a phone.
- Adjust a box: drag a corner. Corners, not edges, because the boxes are quadrilaterals, not axis-aligned rectangles.

Then a single Continue button. If the tool is confident in every detection (all four corners strong, aspect ratio in the check range, no overlaps), the header says so and the operator can Continue on a glance. This step is never skipped automatically, because a wrong count is the error that costs the most downstream.

### 4.3 Reviewing the grid

One long scrolling page. Every check is visible. No paging, no "next check" button, nothing that takes the operator's off the page.

Each check is one row:
- Left: the cropped, straightened, upright check at about 500 px wide, legible without zooming on a laptop. The MICR band along the bottom is blurred by default (a small toggle per row unblurs it, off again on the next batch).
- Right: the fields, one per line, each with a small copy button beside it:
  - Payer (the printed name at the top-left of the check)
  - Amount (the numeric courtesy box)
  - Date
  - Check number (the printed number at the top-right)
  - Memo (the handwritten memo line, often a unit number or "Sept rent")
  - Payee (who the check is written to; expected to be one of a short known list of payees; used as a sanity check, shown but not copied by default)
- A rotate button on the crop (rotates 180 degrees; 90-degree rotations are also available in a small menu) for the case where orientation was guessed wrong. Rotating re-runs the field reads for that row.
- A "done" checkbox at the far right, ticked automatically when the operator copies the row, so a half-finished batch shows the operator's where the operator left off. Done rows dim slightly.

Field states:
- Filled and confident: plain text, editable.
- Filled but unsure: the value is shown with a soft amber highlight and the crop region it came from is outlined when the field has focus. The operator confirms by tabbing past it or edits it.
- Blank: the tool was not confident enough to guess. Empty field with the same amber highlight, placeholder text "read from the check." Clicking the field magnifies the matching region of the crop inline in the row (an inline magnifier, never a modal), so the answer is right there.

Keyboard flow:
- Tab moves through the unsure and blank fields in order, across rows, so a whole batch can be filled without the mouse. Shift+Tab goes back.
- Enter in a field commits it and moves to the next unsure field.
- Ctrl+Z undoes the last edit in a field, including an autocomplete snap the operator did not want.
- Escape closes any open magnifier or lightbox.

Autocomplete for Payer (local only):
- The tool remembers every payer name the operator has confirmed, in localStorage (never cookies, never transmitted). After two typed letters it offers matches. When OCR returns a fuzzy read that is within a small edit distance of a known name, it snaps to the known name and marks the field unsure so the operator sees the snap happened.
- The list is editable from a small settings panel (remove a name, clear all).

Amount handling:
- Normalize to two decimals with no currency symbol, since that pastes cleanly into a numeric spreadsheet column.
- If both the numeric box and the written-out legal line were read and they disagree, show the numeric value marked unsure with a note "the written amount reads differently," never silently pick one.

Date handling:
- Parse into ISO (YYYY-MM-DD) for the copy, display as the operator prefers (a settings toggle: ISO or M/D/YYYY). If the date is blank, offer the email's date as a one-click suggestion when the operator has typed it into a batch-level "email date" box at the top of the review page (optional).

Duplicate warning:
- Check numbers from confirmed past batches are kept in localStorage along with the payer and date. If a check number plus payer matches a past batch, the row shows "Seen before: batch on Sep 12." It is a warning, not a block; managers do occasionally re-send a photo.

### 4.4 Lightbox

Clicking a crop (anywhere outside the buttons) opens it front and center over a dimmed background, sized to fit the screen. X in the corner or Escape closes it, and the operator lands back exactly where the operator was in the scroll. Left and right arrow keys step through the checks without closing. Scroll wheel or pinch zooms; drag pans when zoomed. A rotate button lives here too. The lightbox is the one modal in the app, and it never carries any input fields, so nothing the operator is typing is ever hidden behind it.

### 4.5 Copying out

- Per-field copy buttons, described above, for filling the tracker cell by cell. Each flashes "copied" for a second.
- Per-row copy: copies that row's fields as one tab-separated line, in the column order from settings, and ticks the row's done box.
- "Copy all rows" at the top and bottom of the review page: every row as tab-separated lines, a toast "6 rows copied," all rows ticked done. Row order matches the numbers from the count step so the operator can cross-check against the photo.
- Column order and which columns to include are set once in the settings panel (drag to reorder, checkbox to include) and remembered locally. Default order: Date, Payer, Amount, Check number, Memo. This lets the paste match the operator's tracker's columns exactly without the tool ever seeing the tracker.
- The clipboard text is plain tab-separated values. Pasting into Google Sheets or Excel fills one cell per field with no dialogs.

### 4.6 Finishing

- A "Finish batch" button clears the images from memory, adds the confirmed check numbers and payer names to the local history, and returns to the empty drop zone. A small "6 checks recorded" line stays visible until the next paste.
- If the operator closes the tab or refreshes with unfinished rows, the browser's "leave page?" prompt fires with "You have 6 checks that have not been copied."
- The check images are held in memory only, never written to localStorage or IndexedDB. Closing the tab removes them from the machine. Only text (confirmed payer names, check numbers, dates, settings) persists, and a "Clear everything this page remembers" button in settings wipes that too.

## 5. Processing pipeline

All of this runs in the browser, in a Web Worker so the page never freezes. Stages, in order, with the fallback for each:

1. Decode and orient. Read EXIF orientation, rotate the pixels accordingly, strip EXIF, produce a full-res canvas and a ~2500 px working copy.
2. Find check candidates (classical, no model). On the working copy: grayscale, light blur, adaptive threshold or Canny edges, dilate to close gaps in the check border, find contours, keep those that approximate to four corners with an area above a floor and an aspect ratio in the check range (roughly 2:1 to 2.6:1 in either orientation, tolerating perspective). Checks are high-contrast rectangles on a plain sheet, which is the easy case for this method. Overlapping candidates are merged or the smaller dropped. Output: a list of quadrilaterals in full-res coordinates.
   - Fallback when a photo defeats this (a patterned sheet, heavy shadows): the operator draws the boxes herself in the count step; the tool still does everything after that.
   - If zero candidates are found, the count step says "I could not find any checks in this photo" with a hint (plain background, all four corners in frame) and the Add-a-check tool ready.
3. Rectify. Perspective-warp each quadrilateral from the full-res image to a flat landscape rectangle at a fixed working width (about 1600 px), so every crop is the same shape regardless of how the check lay on the sheet.
4. Orient. A rectified check is either right-side up or upside down (landscape is forced in step 3). Run a fast OCR pass on the top strip and on the same strip rotated 180 degrees; keep whichever yields more confident text. Tie-break with the position of the MICR band, which sits at the bottom of a right-side-up check.
5. Locate fields by layout. US personal and business checks share a layout: payer block top-left, check number top-right, date line right of center near the top, payee line left-middle, courtesy amount box right-middle, legal line below the payee, memo bottom-left, signature bottom-right, MICR band along the bottom. Define each as a region in normalized coordinates with generous margins, then refine each region to the nearest text cluster found by OCR's layout output.
6. Read fields. Run OCR per region rather than on the whole check, which is both faster and more accurate. Printed regions (payer, check number, payee, and sometimes date) are where OCR is reliable. Handwritten regions (amount, date, memo) get the same treatment but with a lower expected confidence.
7. Gate by confidence and cross-checks.
   - Each field carries the OCR confidence for its words. Below a threshold it is marked unsure; below a lower threshold it is left blank. Thresholds are tuned per field on the mock set.
   - Amount: numeric box must parse as a money value; if the legal line also parsed, they must agree; a check number must be all digits and match the top-right print (the MICR check number can be used as a second vote without reading the rest of the band).
   - Date: must parse as a date within a plausible window (a year either side of today).
   - Payer: snap to known names as described in 4.3.
8. Blur the MICR band on the display copy of each crop.

Model choices for v1:
- OpenCV compiled to WebAssembly for stages 1 to 4 (about 8 to 10 MB, cached).
- Tesseract in WebAssembly with the English data for stages 4 to 6 (about 5 MB, cached). Good on printed text, weak on handwriting; that weakness is why the blank-with-crop fallback exists.
- Optional later: a small handwritten-digits model (a few MB, runs in WebAssembly) dedicated to the courtesy amount box, and, if WebGPU is available, a small vision-language model for the handwritten fields with a larger cached download. Both are additive: the UI does not change, some blanks just become filled-but-unsure.

Where the files come from: the app's own JavaScript and CSS live on GitHub Pages. The OpenCV and Tesseract WebAssembly files and the language data are larger than the 100 MB per-file limit only in the model case, but to keep the Pages repo small they are fetched from a public CDN on first load and cached by a service worker, so later visits are fully offline. The service worker also makes the app work with no network at all after the first visit.

Performance budget on a mid-range laptop: under 3 seconds from paste to the count step for a 12-megapixel photo, under 10 seconds from Continue to a fully populated review grid for six checks. Progress is shown per check during the review-grid build so the page never looks stuck.

## 6. Edge cases

- Glare spot on one check: detection usually still works from the border; the OCR region under the glare returns low confidence and goes blank. The crop shows the glare, the operator reads what the operator can.
- Shadow across the sheet: adaptive thresholding handles gradual shadows; a hard shadow edge can split a check into two candidates, which the merge step catches when the two pieces share long edges.
- Two checks overlapping: the overlap produces one wrong quadrilateral. The operator removes it and draws two boxes; the warp handles the partially occluded one as well as it can.
- A check partly out of frame: detected as a non-rectangular contour and dropped. The operator can draw a box; the crop will be missing an edge.
- Checks in both orientations in one photo (some landscape, some portrait on the sheet): fine, since each quadrilateral is warped and oriented independently.
- A check in a plastic sleeve or with a sticky note: the note may cover a field; that field goes blank.
- Photo of a photo (a screenshot of the email forwarded again): lower resolution, still works if the checks are at least about 400 px wide in the image; otherwise the crops are illegible and the tool says "this photo is too small to read; ask for the original attachment."
- Very tall bedsheet photo with 12 checks: fine; the review page just scrolls longer.
- Non-US check formats: out of scope; layout regions will be off and most fields go blank, but crops and rotation still work.
- Browser without WebAssembly: not a real case on current Chrome or Edge; show a plain message rather than a broken page.

## 7. Privacy and data handling

- Nothing leaves the laptop. There is no server, no analytics, no error reporting, no fonts or scripts loaded from anywhere other than the app's own host and the one CDN used for the WebAssembly files on first load. Content Security Policy locks this down so a future edit cannot accidentally add an outbound call.
- Images live in memory only for the duration of a batch. They are never written to any browser storage and are released on Finish batch or tab close.
- Persistent local data is text only: confirmed payer names, check numbers with payer and date for the duplicate warning, and settings. All of it is visible and deletable from the settings panel.
- The MICR band (routing and account numbers) is never extracted into a field, never stored, and is blurred in every displayed crop by default.
- The trust sentence on the page is literally true and stays true; any change that would make it false is out of bounds for this project.

## 8. Build, hosting and testing

- Plain HTML, CSS and JavaScript, no framework and no build step, so the repo on GitHub Pages is the deployed artifact and stays runnable indefinitely. Modules split by responsibility: input, detection, rectification, OCR, field gating, review UI, lightbox, clipboard, local history, settings.
- Public GitHub repo under Owen's account, GitHub Pages from `main`. The repo contains no real check data of any kind.
- Mock check set for development: print a dozen fake checks (fictional names, addresses, banks, the standard layout, a printed MICR-style line of made-up digits), lay them on a bedsheet in the ways people actually do (grid, rotated, upside down, overlapping, one out of frame), photograph with a phone under a few lighting conditions. This set is the regression suite: detection count, orientation, and field reads are checked against a hand-written answer key.
- Real-photo test: the operator tries it on a real batch on the operator's own laptop and reports what went blank and what read wrong. We never receive the photo; we receive the operator's description or, at most, a redacted screenshot the operator chooses to send.
- Browser test on Windows Chrome and Edge, including a laptop without a discrete GPU.

## 9. Open questions

- Her tracker's exact columns and their order, so the default column settings match it on first use. The operator can set this herself in settings, but a matching default is nicer.
- Whether the manager's photos come from Gmail's preview (Copy image works) or only as downloaded files, and whether any arrive as HEIC.
- Whether checks are always laid on a bedsheet or sometimes on a desk or a patterned surface, which decides how robust detection has to be in v1.
- Whether the co-op or the trust is the usual payee, for the payee sanity check.
- Whether the operator wants the memo field at all, or whether the unit number is what the operator actually records.

## 10. Milestones

1. Skeleton: static page on GitHub Pages, drop zone with all three input doors, EXIF handling, trust sentence, service worker caching. Verified on Windows Chrome and Edge.
2. Detection and count step with numbered outlines and manual add, remove, and corner adjust.
3. Rectification, orientation, and the review grid with crops only (no field reads yet), lightbox, rotate, per-row and copy-all with hand-typed fields. At this point the tool is already useful: the operator gets upright legible crops on one page and types beside them.
4. Field reads with confidence gating, autocomplete, duplicate warning, amount and date handling, MICR blur.
5. Mock-set regression suite and the first real-batch trial with the operator.
6. Optional: handwritten amount model, WebGPU vision model for handwritten fields.
