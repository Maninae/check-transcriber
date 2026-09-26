# js/storage/ — what the page remembers between visits (text only)

| Module | Owns |
|---|---|
| `local_store.js` | The only code that touches `localStorage`. Every key starts with `checkTranscriber.v1.`; `removeEveryOwnedKey` backs "Clear everything". Refuses to store anything but plain JSON data. |
| `batch_history.js` | Confirmed payer names (autocomplete, the payer snap) and past checks `{ checkNumber, payer, date, batchDate }` (the duplicate warning). Written by Finish batch. |

Invariants: text only, never a canvas, data URL or pixel buffer; no cookies, no IndexedDB. Keys: `knownPayerNames`, `checkHistory`, `settings`, `payeeNames`.
