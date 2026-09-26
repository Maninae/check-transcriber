# js/review/ — the review grid (spec 4.3-4.5)

One row per check: the upright crop (the payload, 500 px) with its magnifier, the six fields, Copy row and Done. State lives on `ReviewGrid`; everything else is DOM or pure helpers.

| Module | Owns |
|---|---|
| `review_grid.js` | Coordinator: row state, crops, `receiveFieldReads` (the one path for worker reads, re-reads after Rotate and the test hook), re-gating, Rotate, copy, duplicate warning, email date, magnifier open/close. |
| `review_field_states.js` | Pure field-record rules: confident / unsure / blank / confirmed, Tab stops, touched fields surviving re-reads, copy values (ISO date, two-decimal amount). Unit-tested in Node. |
| `review_field_editing.js` | What the operator does inside a field: edit, commit, confirm on Tab/Enter, magnifier on focus, autocomplete pick, email-date click; the debug description. |
| `review_field_view.js` | DOM for one field line: label, input, copy button, amount note, email-date button. |
| `review_row.js` | DOM for one row: crop frame with box outline, magnifier canvas, fields, duplicate line, Copy row, Done. |
| `field_magnifier.js` | Magnifier region (read box + margin, or the field's usual place) and drawing; box outline placement. |
| `payer_autocomplete.js` | The suggestion list under the payer field. |
| `field_definitions.js` | Field keys and labels, default copy columns, fallback field regions. |
| `field_undo.js` | Per-field Ctrl+Z, including values set from script (`setValue`). |
| `clipboard_rows.js`, `crop_rendering.js`, `lightbox.js` | TSV text; rotation + MICR blur; the one modal. |

Invariants:
- Field reads always go through `fields/field_gating.js` `gateCheckFields`; never set a field state by hand.
- A touched field (edited or confirmed) is never overwritten by a later read; it only takes the new box.
- Boxes are in the DISPLAYED crop's full-resolution pixels. Rotate flips them at once; a re-read returns them in the rotated orientation.
- The magnifier draws from `displayedCropCanvas` (MICR blurred), never from `uprightCropCanvas`.
- `countFieldStates()` + `onFieldStatesChanged` feed the review summary line (batch_flow.js); `renderField` schedules it, once per task.
- Only fields take Tab focus; anything added to a row (reread status, enlarge hint) must stay untabbable, or Tab stops crossing rows break.
- Tab stops are fixed by the gate (unsure/blank and not-yet-read fields); confirming does not remove a stop, so Shift+Tab can come back.
