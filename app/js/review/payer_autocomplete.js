/**
 * Payer autocomplete (spec 4.3): after two typed letters, a small list of the payer names
 * confirmed in past batches drops down under the field. Arrow keys highlight, Enter or a
 * click picks, Escape closes.
 *
 * - It only acts on keys while a suggestion is highlighted, so Tab and Enter keep their
 *   normal field-to-field meaning otherwise (Enter with nothing highlighted commits and moves on).
 * - A pick goes through `onPick(name)`, which sets the value with field_undo.js `setValue`,
 *   so Ctrl+Z restores what the operator had typed.
 * - Must be attached before the field's own Enter handler: it stops the key when it uses it.
 */

function createSuggestionItem(name, onChoose) {
  const item = document.createElement("li");
  item.className = "payer-suggestion";
  item.textContent = name;
  item.setAttribute("role", "option");
  // mousedown, not click: picking must not blur the input first.
  item.addEventListener("mousedown", (event) => {
    event.preventDefault();
    onChoose(name);
  });
  return item;
}

/**
 * `suggestNames(typedText)` returns matching known names; `onPick(name)` applies one.
 * Returns `{ close() }`.
 */
export function attachPayerAutocomplete(inputElement, containerElement, { suggestNames, onPick }) {
  const listElement = document.createElement("ul");
  listElement.className = "payer-suggestions";
  listElement.setAttribute("role", "listbox");
  listElement.hidden = true;
  containerElement.append(listElement);
  let suggestions = [];
  let highlightedIndex = -1;

  function close() {
    listElement.hidden = true;
    listElement.replaceChildren();
    suggestions = [];
    highlightedIndex = -1;
  }

  function highlight(index) {
    highlightedIndex = index;
    [...listElement.children].forEach((item, itemIndex) => {
      item.classList.toggle("payer-suggestion--highlighted", itemIndex === index);
      item.setAttribute("aria-selected", String(itemIndex === index));
    });
  }

  function choose(name) {
    close();
    onPick(name);
  }

  // Only the operator's typing fires "input"; values set from script never open the list.
  inputElement.addEventListener("input", () => {
    suggestions = suggestNames(inputElement.value);
    if (suggestions.length === 0) {
      close();
      return;
    }
    listElement.replaceChildren(...suggestions.map((name) => createSuggestionItem(name, choose)));
    listElement.hidden = false;
    highlight(-1);
  });

  inputElement.addEventListener("keydown", (event) => {
    if (listElement.hidden) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      // Slots run "nothing highlighted", then each suggestion, and wrap around.
      const slotCount = suggestions.length + 1;
      const step = event.key === "ArrowDown" ? 1 : -1;
      highlight(((highlightedIndex + 1 + step + slotCount) % slotCount) - 1);
      return;
    }
    if (event.key === "Enter" && highlightedIndex >= 0) {
      event.preventDefault();
      event.stopImmediatePropagation();
      choose(suggestions[highlightedIndex]);
      return;
    }
    if (event.key === "Escape") {
      event.stopImmediatePropagation();
      close();
      return;
    }
    if (event.key === "Tab") close();
  });

  inputElement.addEventListener("blur", close);
  return { close };
}
