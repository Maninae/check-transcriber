# UX audit: Check Transcriber (Sep 26 2026, before the polish pass)

Screenshot-grounded, graded against the 20 foundations (Rams R1-R10, Nielsen N1-N10). Driven with Playwright through the real flow on eval scene `synth/v1.1-closeup-eval/eval_000001.jpg` (six checks on a wood-grain table, two upside down; the detector found five), at 1366x768 (her laptop) and 1920x1080, CPU throttled 4x during processing to approximate a mid-range Windows laptop. Screenshots: `/tmp/check-transcriber-ux/before/<size>/`. A status-text timeline was recorded every 100 ms.

The operator is non-technical, on Chrome or Edge, and should never have to wonder whether the page is working. Owen's brief: "make it feel really intuitive and natural without them having to wonder if it's buggy."

## Moment by moment

| Moment | What she sees (before) | What she might wonder |
|---|---|---|
| First visit, loading | Drop zone; one gray line under the trust sentence: "Getting ready for the first time (about 46 MB, once)." No motion, no progress. | "Is it doing anything? Can I paste yet?" (She can: the photo waits, silently, with "Getting ready…".) |
| Return visit / offline reload | The same "for the first time (about 46 MB, once)" line (`23_offline_reload_early`). | "Why is it downloading again? Did it forget?" The copy is false on a return visit. |
| Dragging a photo over the page | Only the drop zone reacts (`03_drag_over`); dropping anywhere else does nothing (the browser may even open the image in the tab and lose the page). | "Where do I let go?" |
| The instant a photo lands | The drop zone collapses to "Photo received: 3152 x 2364" and a gray "Finding checks…" line (`04`, `05`). No thumbnail, nothing moves. Pixel dimensions are meaningless to her. | "Did it take my photo? Is it stuck?" For 3.5 s (throttled) the page is 80% empty and static. |
| Finding checks | "Finding checks…" then "Tightening the outlines…", each static text. | "Stuck?" No elapsed-time reassurance if a big photo takes long. |
| Count step | "Found 5 checks", a note, Add a check / Continue, the photo at full column width (`06`). At 1366x768 only the top half of the photo is visible: checks 4 and 5 and the missed sixth are below the fold. Badge 1 covers the payer name. | "How many are there really?" She cannot count against the photo without scrolling. "What do the green and amber outlines mean?" (the note mentions amber only when some are amber). |
| Selecting / adding | White outline, red X, handles (`07`); Add a check toggles a crosshair with no instruction text (`08`). | "Now what do I do? Drag? Click corners?" |
| Continue pressed | The page smooth-scrolls to the grid; the only progress line is left off-screen at the top (`09`). Rows show plain empty inputs before reads arrive, identical to "nothing was found". | "Did it read nothing? Is it broken?" No sign the reads are still coming. Continue can be pressed again while straightening (the count step locks, but there is no busy state on the button). |
| Grid complete | Amber and plain fields, "read from the check" placeholders (`11`). No "all read" moment. | "Is it finished? What does amber mean? Why is 'Richard Salinas' amber when it is right?" |
| Field focus | Box outline on the crop and inline magnifier (`12`): works well. | none |
| Rotate | Crop flips instantly; the fields keep the old values until the re-read lands, with no sign a re-read is happening (`13`). | "Did rotating do anything to the fields?" |
| Copy field / row | The row's "Copy row" flashes "Copied", row dims to 55% (`15`); the copy icon is a faint ⧉ glyph at 2.6:1 contrast. | "Can I click that little symbol?" |
| Copy all | "5 rows copied" black pill, bottom centre, no icon (`16`). Lingers over the page after Finish (`18`). | none, mostly fine |
| Settings | Inline panel above the page, three columns, no close button besides the header link (`17`). Empty payee list is invisible (just an input). The handwriting blurb is a four-sentence paragraph. | "How do I close this?" "What is a payee for?" |
| Handwriting switch | Status text only; the 128 MB download reports "Downloading…: 12 of 128 MB" as text, no bar. (The network-blocked capture was invalid in this run: the page-level route did not intercept the worker's fetch.) | "How long will this take?" |
| Finish batch | Back to the empty drop zone with a small gray "5 checks recorded" (`18`). | Understated; no "ready for the next photo" closure. |
| Unsupported file | Rose box: "That doesn't look like a photo this tool can read. Use a JPEG, PNG, or WebP image." (`19`) | Says what, not how to get the right file. |
| HEIC | Rose box with the Gmail Copy image hint (`20`). Good. | none |
| Zero checks | Serif heading "I could not find any checks in this photo", tips, Add a check already pressed (`21`). Reasonable, but "Add a check" being pre-pressed is not explained ("drag a box around each one" is buried in the note). | "Do I click the button or drag?" |
| Engine failure (first visit, offline) | Gray line + Retry under the trust sentence (`25`); the drop zone still invites a paste that can never be processed. | "Is it my internet? Can I still paste?" |

## Measured

- Continue on the count step: bottom at 267 px of 768, above the fold (pass). The photo frame starts at 279 px and is 849 px tall: 61% of the photo is below the fold at 1366x768.
- Timeline (throttled): paste to count step ~3.5 s over two static texts; Continue to full grid ~2.5 s; each stage text shows under one second, so a per-check text alone reads as flicker, not progress.
- WCAG contrast failures: "read from the check" placeholder #b39462 on #fcefd6 = 2.52:1 (needs 4.5); inactive step labels #aaa092 on #f6f2ea = 2.31:1; copy icon #aaa092 on white = 2.58:1 (UI needs 3:1); amber field border #d99a2b on white = 2.44:1 (UI 3:1); amber text #9a6414 on the page background = 4.47:1 (marginal).
- No `:focus-visible` style on any button (buttons fall back to the browser default ring); fields have a 2px ring. `prefers-reduced-motion` is not referenced anywhere (the only motion is the scroll-into-view and 150 ms fades).

## Scorecard

| # | Principle | Verdict | Evidence |
|---|---|---|---|
| R1 | Innovative | PASS | Paste-anywhere, inline magnifier over the crop, direct-manipulation quads: uses the medium well (`12`, `07`). |
| R2 | Useful | FLAG | "Photo received: 3152 x 2364" pixel dimensions serve no job of hers (`04`). |
| R3 | Aesthetic | PASS | Warm paper palette, serif display heading, one accent; calm and deliberate (`02`, `11`). |
| R4 | Understandable | FLAG | Amber/plain meaning never stated; the count outlines' green/amber unexplained (`06`, `11`). |
| R5 | Unobtrusive | PASS | Chrome is small and gray; photo and crops are the payload. |
| R6 | Honest | FAIL | Unread rows look identical to "nothing found" (`09`, `10`); "for the first time, 46 MB" shown on return visits (`23`). |
| R7 | Long-lasting | PASS | No trend skinning. |
| R8 | Thorough to the detail | FLAG | Five contrast failures above; no focus-visible styling on buttons; badge 1 covers the payer name (`06`); toast lingers over the finished page (`18`). |
| R9 | Frugal | PASS | Nothing polls; download size is stated before the handwriting download. |
| R10 | As little design as possible | PASS | Few elements; nothing decorative. |
| N1 | Visibility of system status | FAIL | Static text during 3.5 s of detection, progress line scrolled off-screen during reads (`09`), silent Rotate re-read (`13`), no completion moment. The core of Owen's complaint. |
| N2 | Real-world match | FLAG | "Photo received: W x H"; "Tightening the outlines"; "Payees" needs its explanation. Otherwise plain words, and no "AI"/"model" anywhere (grep clean). |
| N3 | User control and freedom | FLAG | Settings has no close control in the panel; Start over present; Ctrl+Z per field present. |
| N4 | Consistency | FLAG | Two feedback styles for copy (✓ in the icon, "Copied" on the row, toast for all); errors live in three places (rose hint box, gray engine line, progress line). |
| N5 | Error prevention | FLAG | Continue has no busy state; dropping outside the drop zone lets the browser navigate away from an open batch (loses the work). |
| N6 | Recognition over recall | FLAG | The Gmail paste recipe, the count step's gestures and the Tab shortcut exist only in her memory or the spec (`02`, `08`). |
| N7 | Flexibility and efficiency | PASS | Tab/Enter/Ctrl+Z, copy all, autocomplete. |
| N8 | Minimalist | PASS | Information density is right; the handwriting paragraph is the one long block. |
| N9 | Error recovery | FLAG | Unsupported-file and engine-failure messages say what, not what to do next; engine failure leaves the drop zone inviting a paste. |
| N10 | Help | FLAG | No contextual hints at all; spec forbids a tour, so the answer is gentle, dismissible first-run hints. |

## Ranked gaps

- **P0 (reads as broken)**
  1. Silent processing (N1, R6): after paste, static text on a nearly empty page; after Continue, the progress line off-screen and unread rows indistinguishable from empty reads. Fix: a moving progress panel pinned where she is looking (under the drop zone after paste; sticky at the top of the grid after Continue) with plain stage copy, a determinate bar driven by the per-check callbacks, an elapsed-time reassurance after a few seconds, and skeleton shimmer in every unread field and crop.
  2. Whole count photo not visible at 1366x768 (R4, N6): fit the photo to the viewport height below the header so every outline is visible at once.
  3. "Getting ready for the first time (about 46 MB, once)" on return and offline visits (R6): only say "first time" and the size when the page is not already served by the offline cache.
- **P1 (confusing)**
  4. Photo acknowledgement: no thumbnail, pixel dimensions (N1, R2): fade in a small canvas thumbnail and "Photo received" the instant it lands.
  5. Page-wide drag target (N5): the whole page highlights "Drop the photo anywhere" and accepts the drop, so a stray drop never navigates away.
  6. Field-state meaning (R4): one dismissible line above the grid: "Plain = read confidently. Amber = please check. Blank = read it from the check."
  7. Contrast failures (R8, AA): placeholder, copy icon, amber borders, inactive steps, amber text.
  8. Rotate gives no feedback (N1): the row's fields go back to shimmering until the re-read lands.
  9. Errors that say what to do (N9): unsupported file, zero checks, engine failure (disable the drop zone's invitation and say "you can still paste once it loads"), mid-batch failures with a Start over button.
- **P2 (friction)**
  10. First-run hints (N6, N10): paste-from-Gmail recipe on the empty drop zone; count-step gestures; copy row vs copy all and the Tab shortcut in the grid. Dismissible, stored locally, never modal.
  11. Busy and disabled states (N5, R6): Continue shows "Straightening…" and cannot be pressed twice; disabled Continue says why on hover.
  12. Completion moments (N1): "All 5 checks read" when the grid is done; Finish shows "5 checks recorded. Ready for the next photo."
  13. Toast (N4): icon, consistent position, cleared on Finish.
  14. Settings (N3, N8): a close button, one-line explanations per group, an empty-state line for payees, the handwriting download as a bar.
- **P3 (polish)**
  15. Visible `:focus-visible` rings on every button; `prefers-reduced-motion` honoured.
  16. Count badges: stronger ring for busy photos; do not overlap the payer name when avoidable.
  17. Copy icon: a drawn two-squares icon at 3:1+ instead of a faint glyph.

## SOLID (do not re-litigate)

Trust sentence placement and wording; the calm palette and serif display face; the inline magnifier; direct-manipulation quads; Tab walking only review stops; HEIC hint copy; no "AI"/"model" words in the UI.

## Not covered

Real Windows rendering (Segoe UI metrics, ClearType, Windows High Contrast mode); real Gmail drag and clipboard behaviour (simulated with synthetic events); touch screens; screen-reader walkthrough beyond aria attributes; the handwriting download failure in the before run (page-level route missed the worker's fetch; re-captured in the after run with the context offline).

## After the polish pass (branch app/ux-polish)

Screenshots: `/tmp/check-transcriber-ux/after/<size>/`, same driver and scene. Timeline and layout metrics in `timeline.json` beside them.

| Gap | What closed it |
|---|---|
| 1. Silent processing | Progress panel (`js/progress_panel.js`): animated scan glyph, plain stage copy (`js/progress_copy.js`: "Looking for checks…", "Straightening check 3 of 5…", "Reading the printed fields…", "Reading handwriting… (this takes a moment)"), a determinate bar during Continue (checks straightened + read), a sliding bar when not countable, "Still working, large photo" after 4 s, "All 5 checks read" on completion. It moves to the top of the review step on Continue. Unread fields and crops shimmer; values fade in. |
| 2. Count photo below the fold | `count_step.js fitPhotoFrameToWindow`: at 1366x768 the whole photo and every outline are visible after the step scrolls into view; Continue's bottom edge sits at 84 px. |
| 3. "First time, 46 MB" on return visits | Readiness copy checks `navigator.serviceWorker.controller`; return and offline visits say "Getting ready…". |
| 4. Photo acknowledgement | "Opening the photo…" the instant a file lands, then a fading-in thumbnail and "Photo received"; the photo itself appears at once, dimmed with a light sweeping across it, exactly where the outlines will land. A small-preview warning shows only when the long side is under 1500 px. |
| 5. Page-wide drag target | Whole-page overlay "Drop the photo anywhere"; drops anywhere are taken in, so a near miss never navigates away. |
| 6. Field-state meaning | "What the colours mean, and shortcuts" hint with miniature fields (plain / amber / empty amber), plus the review summary "21 fields need a look (amber)" and a tooltip on every amber field. |
| 7. Contrast | Placeholder 5.2:1, amber outline 3.3:1, amber text 5.2:1, copy icon 4.1:1, step labels 5.0:1. |
| 8. Rotate feedback | "Reading this check again…" and shimmering untouched fields until the re-read lands; an info toast if it fails. |
| 9. Errors | Each says what to do: unsupported file (with the Gmail recipe), undecodable photo, zero checks ("I couldn't find any checks. Try a photo on a plainer background… or add them by hand"), engine failure card with a primary Retry, mid-batch failures with a Start over button, a photo waiting on failed tools says so. |
| 10. First-run hints | Three `<details>` hints, auto-open for 3 sessions unless "Got it", then a one-line link (`js/first_run_hints.js`, stored through LocalStore). |
| 11. Busy and disabled states | Continue becomes a spinner "Straightening…" and ignores a second press; disabled Continue says why on hover; the count note says what to do right now (drawing, a check selected). |
| 12. Completion moments | "All N checks read" in the panel; a success card "N checks recorded. Ready for the next photo." after Finish. |
| 13. Toast | One style, tick or info icon, bottom centre, time scales with length; Copy row adds "Check 2 copied. Paste it into your tracker with Ctrl+V." |
| 14. Settings | Title row with Close (and Esc), one-line explanation per group, empty payee state, download bar, red failure text, fixed three columns. |
| 15-17. Polish | `:focus-visible` rings, reduced motion honoured, badges on the check's top edge with a white ring, SVG copy icon with a tick flash, numbered step discs with ticks for finished steps. |

Found while verifying: the count-step note changed height at pointer-down (the drawing instructions only appeared mid-drag), shifting the photo 21 px under the pointer; the milestone refit test caught it (18.7 px error vs 0.0 on main). The note now updates when the mode changes and reserves two lines.

### Second round (fresh Opus reviewer, no P0)

- P1: engines failing with a pasted photo is one panel message with Retry ("Your photo is kept"), and Retry continues that photo; HEIC and non-photos are refused before any question, as an info toast when a batch is open; the photo shows dimmed and scanning in the count step's place while the checks are found, then brightens and the outlines fade in 120 ms apart, with no layout or scroll jump (measured: photo top 220 px before and after).
- P2: numbers sit just outside each check's top edge; window.confirm is replaced by an inline bar in the photo strip ("Replace these 4 checks with the new photo?" Keep working / Replace), holding the new file; settings lay out by the panel's own width (container query), Handwriting full width under 900 px; shorter count notes.
- P3: one done message; toasts cleared on Finish and Start over; "Photo received" without pixel sizes; outlines keyboard-reachable.
- Resolved ambiguity: the regression tests accepted window.confirm through dialog handlers. Their files and assertions are unchanged; the shared `paste_image_file` helper answers the inline bar instead, and a replaced batch's timing starts when she answers.
