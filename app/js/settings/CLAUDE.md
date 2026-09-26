# js/settings/ — the operator's settings and the settings panel

| Module | Owns |
|---|---|
| `app_settings.js` | The settings model: date display, copy columns `[{ key, included }]`, the payee list, the handwriting-reader switch. Saves through `storage/local_store.js`; `onChange(listener)` tells the grid what changed. |
| `settings_panel.js` | The inline panel (not a modal): fills the static DOM in index.html, handles every control, the handwriting switch's progress/failure text, and "Clear everything". |
| `column_order_list.js` | The draggable "Columns to copy" list (Alt+Up/Down from the keyboard). |
| `settings_panel_setup.js` | DOM lookups and wiring to the grid and the pipeline client, so main.js stays small. |

Invariants:
- The handwriting switch is off by default and only the operator turns it on; its state persists and is re-applied once the engines are ready.
- The UI copy never says "AI" or "model"; the switch is "Read handwriting (one-time 128 MB download)".
