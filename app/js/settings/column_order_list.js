/**
 * The "Columns to copy" list in the settings panel (spec 4.5): one line per field with a
 * checkbox to include it, dragged up or down to set the paste order (Alt+Up / Alt+Down on
 * the checkbox does the same from the keyboard).
 *
 * Stateless over the list it is given: every change calls `onChange(newColumns)` with the
 * whole `[{ key, included }]` list; the caller saves it and calls `render` again.
 */

import { FIELD_LABELS } from "../review/field_definitions.js";

const DRAG_HANDLE_GLYPH = "⠿"; // six dots, the usual "grab here" mark

function moveColumn(columns, fromIndex, toIndex) {
  const reordered = columns.map((column) => ({ ...column }));
  const [moved] = reordered.splice(fromIndex, 1);
  reordered.splice(toIndex, 0, moved);
  return reordered;
}

/** Where a drop at `clientY` over `item` inserts: before it (upper half) or after it. */
function dropIndexFor(item, clientY) {
  const itemRect = item.getBoundingClientRect();
  const itemIndex = Number(item.dataset.columnIndex);
  return clientY < itemRect.top + itemRect.height / 2 ? itemIndex : itemIndex + 1;
}

function createColumnItem(column, columnIndex, columns, onChange) {
  const item = document.createElement("li");
  item.className = "copy-column";
  item.draggable = true;
  item.dataset.columnKey = column.key;
  item.dataset.columnIndex = String(columnIndex);
  const handle = document.createElement("span");
  handle.className = "copy-column-handle";
  handle.textContent = DRAG_HANDLE_GLYPH;
  handle.setAttribute("aria-hidden", "true");
  const label = document.createElement("label");
  label.className = "copy-column-label";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = column.included;
  checkbox.title = "Alt+Up or Alt+Down moves this column";
  checkbox.addEventListener("change", () => {
    onChange(columns.map((other, otherIndex) => (otherIndex === columnIndex ? { ...other, included: checkbox.checked } : { ...other })));
  });
  checkbox.addEventListener("keydown", (event) => {
    if (!event.altKey || (event.key !== "ArrowUp" && event.key !== "ArrowDown")) return;
    event.preventDefault();
    const targetIndex = columnIndex + (event.key === "ArrowUp" ? -1 : 1);
    if (targetIndex < 0 || targetIndex >= columns.length) return;
    onChange(moveColumn(columns, columnIndex, targetIndex), column.key);
  });
  label.append(checkbox, document.createTextNode(FIELD_LABELS[column.key]));
  item.append(handle, label);
  return item;
}

export class ColumnOrderList {
  /** `listElement`: the <ol>; `onChange(newColumns, keyToRefocus?)`. */
  constructor(listElement, onChange) {
    this.listElement = listElement;
    this.onChange = onChange;
    this.columns = [];
    this.draggedIndex = null;
    listElement.addEventListener("dragstart", (event) => {
      const item = event.target.closest(".copy-column");
      if (!item) return;
      this.draggedIndex = Number(item.dataset.columnIndex);
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", item.dataset.columnKey);
      item.classList.add("copy-column--dragging");
    });
    listElement.addEventListener("dragover", (event) => {
      if (this.draggedIndex === null || !event.target.closest(".copy-column")) return;
      event.preventDefault(); // allows the drop
      event.dataTransfer.dropEffect = "move";
    });
    listElement.addEventListener("drop", (event) => {
      const item = event.target.closest(".copy-column");
      if (this.draggedIndex === null || !item) return;
      event.preventDefault();
      const insertIndex = dropIndexFor(item, event.clientY);
      // Removing the dragged line first shifts every later line up by one.
      const targetIndex = insertIndex > this.draggedIndex ? insertIndex - 1 : insertIndex;
      const fromIndex = this.draggedIndex;
      this.draggedIndex = null;
      if (targetIndex !== fromIndex) this.onChange(moveColumn(this.columns, fromIndex, targetIndex));
    });
    listElement.addEventListener("dragend", () => {
      this.draggedIndex = null;
      listElement.querySelectorAll(".copy-column--dragging").forEach((item) => item.classList.remove("copy-column--dragging"));
    });
  }

  /** Shows `columns`; `keyToRefocus` keeps keyboard focus on a line that just moved. */
  render(columns, keyToRefocus = null) {
    this.columns = columns;
    this.listElement.replaceChildren(...columns.map((column, index) => createColumnItem(column, index, columns, this.onChange)));
    if (keyToRefocus) {
      const checkbox = this.listElement.querySelector(`[data-column-key="${keyToRefocus}"] input`);
      if (checkbox) checkbox.focus();
    }
  }
}
