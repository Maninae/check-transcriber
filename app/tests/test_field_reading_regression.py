"""Field reading in the browser vs the Python reference, on 30 eval check crops.

    /Volumes/vega/datasets/check-transcriber/venv/bin/python app/tests/test_field_reading_regression.py [--handwriting]

Each crop (from synth v1's OCR eval manifest, decoded with cv2 so both sides see identical
pixels) goes through the real pipeline worker (`readFieldsOfCrop`) and the real gate
(js/fields/field_gating.js) in headless Chromium, and through field_reading/python_field_reader.py
+ python_field_gate.py. Reported:
- parity: identical boxes, texts, readers; max confidence difference; gate state/value agreement
  (the JS gate vs the Python gate mirror, both on the browser's reads);
- accuracy vs ground truth (experiments' `normalize_field_value`): raw read exact-match per field,
  fill rate at the gate (confident share) and the accuracy of what the gate fills;
- per-crop read time in the browser.
`--handwriting` also turns the opt-in handwriting reader on (both sides; the browser downloads it
once into a persistent profile on vega, then reads it from the service-worker cache) and reports
its parity and per-crop latency. Exits non-zero if parity fails.
"""

import argparse
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from browser_test_helpers import serve_directory, wait_for_engines_ready  # noqa: E402
from experiments.field_reading.data_access.field_manifest import load_field_rows  # noqa: E402
from experiments.field_reading.metrics.field_value_parsing import canonical_value_for_field, normalize_field_value  # noqa: E402
from field_reading.python_field_gate import gate as python_gate  # noqa: E402
from field_reading.python_field_reader import FIELD_NAMES, PythonFieldReader, PythonHandwritingReader  # noqa: E402

CROP_COUNT = 30
SYNTH_V1_MANIFEST = Path("/Volumes/vega/datasets/check-transcriber/synth/v1/manifest.json")
STORED_PREDICTIONS = Path("/Volumes/vega/datasets/check-transcriber/field-reading/predictions/eval/crnn_amount_route__loc=segnet_mobilenetv3l_768.jsonl")
# Test-only "today" for the plausible-date window: the median synthetic eval date, so the window
# (a year either side) tests the gate rather than the calendar. The app uses the real today.
REGRESSION_TODAY_ISO = "2026-03-31"
PERSISTENT_PROFILE_DIRECTORY = Path("/Volumes/vega/datasets/check-transcriber/app-test-profiles/handwriting-reader")
OUTPUT_DIRECTORY = Path("/tmp/check-transcriber-m4")
TIMEOUT_MS = 300_000
GATE_KEY_BY_FIELD = {"payer_name": "payer", "payee": "payee", "amount_numeric": "amount", "date": "date",
                     "memo": "memo", "check_number": "checkNumber"}
CONFIDENCE_TOLERANCE = 1e-3
BOX_TOLERANCE_PX = 1e-6

READ_IN_PAGE_SCRIPT = """
async ({ base64Rgba, width, height, enableHandwriting, knownPayeeNames, todayIso }) => {
  const { loadProcessingEngines } = await import('./js/engine_loader.js');
  const { gateCheckFields } = await import('./js/fields/field_gating.js');
  const pipelineClient = await new Promise((resolve, reject) => loadProcessingEngines((engines) => resolve(engines.pipelineClient), reject));
  if (enableHandwriting && !window.__handwritingEnabled) {
    await pipelineClient.setHandwritingReaderEnabled(true, (text) => console.log(text));
    window.__handwritingEnabled = true;
  }
  const bytes = Uint8Array.from(atob(base64Rgba), (character) => character.charCodeAt(0));
  const startedAt = performance.now();
  const { rawReads, timingsMs } = await pipelineClient.readFieldsOfCrop(width, height, bytes.buffer);
  const elapsedMs = performance.now() - startedAt;
  const gated = gateCheckFields(rawReads, { knownPayerNames: [], knownPayeeNames, todayIso });
  return { rawReads, gated, elapsedMs, timingsMs };
}
"""


def select_eval_crops() -> list[dict]:
    """CROP_COUNT checks evenly spaced through the eval OCR manifest, with their ground truth rows."""
    rows = load_field_rows("eval")
    rows = rows[rows.field_name.isin(FIELD_NAMES)]
    check_paths = sorted(rows.check_crop_path.unique())
    picked = [check_paths[round(index * (len(check_paths) - 1) / (CROP_COUNT - 1))] for index in range(CROP_COUNT)]
    return [{"check_crop_path": path, "truth_rows": rows[rows.check_crop_path == path]} for path in picked]


def load_stored_predictions() -> dict[str, str]:
    """row_key -> pred_text from the research pipeline's own (batched, MPS) eval run."""
    stored = {}
    for line in STORED_PREDICTIONS.read_text().splitlines():
        row = json.loads(line)
        stored[row["row_key"]] = row["pred_text"]
    return stored


def is_read_correct(field_name: str, text: str, truth_row) -> bool:
    """experiments' definition of a correct read (money / date / digits / normalized text)."""
    predicted = normalize_field_value(field_name, text or "")
    canonical_column = {"amount_numeric": "canonical_amount_cents", "amount_words": "canonical_amount_cents",
                        "date": "canonical_date_iso", "check_number": "canonical_check_number"}.get(field_name)
    truth = canonical_value_for_field(field_name, getattr(truth_row, canonical_column)) if canonical_column else None
    if truth is None:
        truth = normalize_field_value(field_name, truth_row.text)
    return predicted is not None and predicted == truth


def is_gated_value_correct(gate_key: str, value: str, truth_rows) -> bool:
    """Whether a filled (copy-ready) value matches the check's ground truth."""
    field_name = {v: k for k, v in GATE_KEY_BY_FIELD.items()}[gate_key]
    truth_row = truth_rows[truth_rows.field_name == field_name]
    if truth_row.empty:
        return False
    truth_row = truth_row.iloc[0]
    if gate_key == "payee":
        return value == truth_row.canonical_payee
    return is_read_correct(field_name, value, truth_row)


def compare_reads(browser_reads: dict, python_reads: dict, report: dict) -> None:
    """Accumulate per-field parity between browser and Python raw reads."""
    for field_name in FIELD_NAMES:
        browser, python = browser_reads.get(field_name), python_reads.get(field_name)
        entry = report.setdefault(field_name, {"fields": 0, "box_same": 0, "text_same": 0, "reader_same": 0,
                                                "max_confidence_difference": 0.0, "max_printed_confidence_difference": 0.0,
                                                "max_style_difference": 0.0, "mismatches": []})
        entry["fields"] += 1
        if browser is None or python is None:
            same = browser is None and python is None
            entry["box_same"] += same
            entry["text_same"] += same
            entry["reader_same"] += same
            if not same:
                entry["mismatches"].append({"browser": browser, "python": python})
            continue
        entry["box_same"] += max(abs(a - b) for a, b in zip(browser["box"], python["box"])) <= BOX_TOLERANCE_PX
        entry["text_same"] += browser["text"] == python["text"]
        entry["reader_same"] += browser["reader"] == python["reader"]
        entry["max_confidence_difference"] = max(entry["max_confidence_difference"], abs(browser["confidence"] - python["confidence"]))
        if browser["reader"] != "trocr":
            entry["max_printed_confidence_difference"] = max(entry["max_printed_confidence_difference"], abs(browser["confidence"] - python["confidence"]))
        entry["max_style_difference"] = max(entry["max_style_difference"], abs(browser["handwrittenProbability"] - python["handwrittenProbability"]))
        if browser["text"] != python["text"]:
            entry["mismatches"].append({"reader": browser["reader"], "browser": browser["text"], "python": python["text"],
                                        "browser_confidence": browser["confidence"], "python_confidence": python["confidence"]})


def is_handwriting_mismatch(mismatch: dict) -> bool:
    """A text mismatch on a read the handwriting reader made (see the parity note in main)."""
    return mismatch.get("reader") == "trocr"


def main() -> None:
    """Run both sides, compare, print and save the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handwriting", action="store_true", help="also turn the opt-in handwriting reader on")
    arguments = parser.parse_args()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    known_payee_names = json.loads(SYNTH_V1_MANIFEST.read_text())["split_pools"]["eval"]["payee_names"]
    crops = select_eval_crops()
    python_reader = PythonFieldReader(PythonHandwritingReader() if arguments.handwriting else None)
    # Report-only: the research recipe (cv2 INTER_LINEAR through this Mac's ARM HAL, which no browser reproduces).
    research_recipe_reader = PythonFieldReader(research_line_resize=True)
    research_recipe_agreement = {"fields": 0, "same_text": 0}
    stored_predictions = load_stored_predictions()

    parity, gate_agreement, per_crop_ms = {}, {"fields": 0, "same": 0, "mismatches": []}, []
    handwriting_ms_per_read = []
    accuracy = {name: {"rows": 0, "browser_correct": 0, "stored_correct": 0, "stored_same_text": 0} for name in FIELD_NAMES}
    fill = {key: {"checks": 0, "confident": 0, "unsure": 0, "blank": 0, "confident_correct": 0, "unsure_correct": 0} for key in GATE_KEY_BY_FIELD.values()}
    serving = serve_directory()
    base_url = serving.__enter__()
    with sync_playwright() as playwright:
        if arguments.handwriting:
            PERSISTENT_PROFILE_DIRECTORY.mkdir(parents=True, exist_ok=True)
            context = playwright.chromium.launch_persistent_context(str(PERSISTENT_PROFILE_DIRECTORY), headless=True)
            page = context.new_page()
        else:
            browser = playwright.chromium.launch()
            page = browser.new_page()
        page.on("pageerror", lambda error: print(f"page error: {error}"))
        page.goto(base_url)
        wait_for_engines_ready(page, TIMEOUT_MS)
        for crop_number, crop in enumerate(crops, 1):
            crop_rgb = cv2.cvtColor(cv2.imread(crop["check_crop_path"]), cv2.COLOR_BGR2RGB)
            height, width = crop_rgb.shape[:2]
            rgba = np.concatenate([crop_rgb, np.full((height, width, 1), 255, np.uint8)], axis=2)
            started = time.perf_counter()
            python_reads = python_reader.read(crop_rgb)
            python_seconds = time.perf_counter() - started
            result = page.evaluate(READ_IN_PAGE_SCRIPT, {
                "base64Rgba": base64.b64encode(rgba.tobytes()).decode(), "width": width, "height": height,
                "enableHandwriting": arguments.handwriting, "knownPayeeNames": known_payee_names, "todayIso": REGRESSION_TODAY_ISO})
            per_crop_ms.append(result["elapsedMs"])
            handwriting_read_count = sum(1 for read in result["rawReads"].values() if read and read["reader"] == "trocr")
            if handwriting_read_count and "handwriting" in result["timingsMs"]:
                handwriting_ms_per_read.append(result["timingsMs"]["handwriting"] / handwriting_read_count)
            compare_reads(result["rawReads"], python_reads, parity)
            research_reads = research_recipe_reader.read(crop_rgb)
            for field_name in FIELD_NAMES:
                browser_read, research_read = result["rawReads"].get(field_name), research_reads.get(field_name)
                research_recipe_agreement["fields"] += 1
                research_recipe_agreement["same_text"] += (browser_read or {}).get("text") == (research_read or {}).get("text")
            # The gate logic is compared on the SAME reads (the browser's), so reader drift (the
            # handwriting reader's int8 decoder, native vs WASM) cannot hide or fake a gate difference.
            python_gated = python_gate(result["rawReads"], [], known_payee_names, REGRESSION_TODAY_ISO)
            truth_rows = crop["truth_rows"]
            for gate_key, (python_state, python_value) in python_gated.items():
                browser_field = result["gated"][gate_key]
                gate_agreement["fields"] += 1
                same = browser_field["state"] == python_state and browser_field["value"] == python_value
                gate_agreement["same"] += same
                if not same:
                    gate_agreement["mismatches"].append({"check": crop["check_crop_path"], "field": gate_key,
                                                         "browser": [browser_field["state"], browser_field["value"]],
                                                         "python": [python_state, python_value]})
                field_name = {v: k for k, v in GATE_KEY_BY_FIELD.items()}[gate_key]
                if truth_rows[truth_rows.field_name == field_name].empty:
                    continue
                tally = fill[gate_key]
                tally["checks"] += 1
                tally[browser_field["state"]] += 1
                if browser_field["state"] != "blank" and is_gated_value_correct(gate_key, browser_field["value"], truth_rows):
                    tally[f"{browser_field['state']}_correct"] += 1
            for truth_row in truth_rows.itertuples():
                read = result["rawReads"].get(truth_row.field_name)
                entry = accuracy[truth_row.field_name]
                entry["rows"] += 1
                entry["browser_correct"] += is_read_correct(truth_row.field_name, read["text"] if read else "", truth_row)
                stored_text = stored_predictions.get(truth_row.row_key, "")
                entry["stored_correct"] += is_read_correct(truth_row.field_name, stored_text, truth_row)
                entry["stored_same_text"] += (read["text"] if read else "") == stored_text
            print(f"[{crop_number:2d}/{len(crops)}] {Path(crop['check_crop_path']).name}: browser {result['elapsedMs']:.0f} ms "
                  f"(localize {result['timingsMs']['localization']:.0f}, read {result['timingsMs']['reading']:.0f}), python {python_seconds * 1000:.0f} ms", flush=True)
        page.close()
    serving.__exit__(None, None, None)

    print("\nParity, browser vs Python reference (same pixels):")
    parity_ok = True
    for field_name, entry in parity.items():
        print(f"  {field_name:15s} boxes {entry['box_same']}/{entry['fields']}  texts {entry['text_same']}/{entry['fields']}  readers {entry['reader_same']}/{entry['fields']}"
              f"  max |conf diff| printed {entry['max_printed_confidence_difference']:.2e}, any {entry['max_confidence_difference']:.2e}"
              f"  max |p_hw diff| {entry['max_style_difference']:.2e}")
        # Handwriting-reader reads are compared but not failed on: its int8 decoder's quantized kernels
        # differ between native and WASM onnxruntime (with the fp32 decoder both sides match exactly).
        printed_mismatches = [m for m in entry["mismatches"] if not is_handwriting_mismatch(m)]
        parity_ok &= not printed_mismatches and entry["box_same"] == entry["fields"] and entry["reader_same"] == entry["fields"]
        parity_ok &= entry["max_printed_confidence_difference"] < CONFIDENCE_TOLERANCE
        for mismatch in entry["mismatches"][:3]:
            print(f"      mismatch{' (handwriting reader, reported only)' if is_handwriting_mismatch(mismatch) else ''}: {mismatch}")
    print(f"  texts identical to the research recipe (cv2 ARM-HAL INTER_LINEAR resize; report only): "
          f"{research_recipe_agreement['same_text']}/{research_recipe_agreement['fields']}")
    print(f"  gate state+value agreement (JS gate vs Python gate on the browser's reads): {gate_agreement['same']}/{gate_agreement['fields']}")
    for mismatch in gate_agreement["mismatches"][:5]:
        print(f"      {mismatch}")
    parity_ok &= gate_agreement["same"] == gate_agreement["fields"]

    print("\nRaw read exact-match vs ground truth (browser | research eval run on the same rows, batched on MPS):")
    for field_name, entry in accuracy.items():
        if entry["rows"]:
            print(f"  {field_name:15s} {entry['browser_correct']:3d}/{entry['rows']:3d} = {100 * entry['browser_correct'] / entry['rows']:5.1f}%"
                  f"   | stored {100 * entry['stored_correct'] / entry['rows']:5.1f}%   identical text to stored {entry['stored_same_text']}/{entry['rows']}")
    print("\nFill rate at the gate (share of checks carrying the field) and accuracy of what is filled:")
    for gate_key, tally in fill.items():
        if tally["checks"]:
            confident_accuracy = f"{100 * tally['confident_correct'] / tally['confident']:5.1f}%" if tally["confident"] else "   - "
            unsure_accuracy = f"{100 * tally['unsure_correct'] / tally['unsure']:5.1f}%" if tally["unsure"] else "   - "
            print(f"  {gate_key:12s} confident {tally['confident']:2d}/{tally['checks']} ({confident_accuracy} right)"
                  f"  unsure {tally['unsure']:2d} ({unsure_accuracy} right)  blank {tally['blank']:2d}")
    print(f"\nBrowser read time per crop: median {statistics.median(per_crop_ms):.0f} ms, max {max(per_crop_ms):.0f} ms"
          f"{' (handwriting reader on)' if arguments.handwriting else ''}")
    if handwriting_ms_per_read:
        print(f"Handwriting reader per field crop: median {statistics.median(handwriting_ms_per_read):.0f} ms "
              f"(per-check means; {len(handwriting_ms_per_read)} checks had handwritten fields)")
    suffix = "__handwriting" if arguments.handwriting else ""
    (OUTPUT_DIRECTORY / f"field_reading_regression{suffix}.json").write_text(json.dumps(
        {"parity": parity, "research_recipe_agreement": research_recipe_agreement, "gate_agreement": gate_agreement, "accuracy": accuracy, "fill": fill, "per_crop_ms": per_crop_ms}, indent=1, default=str))
    if not parity_ok:
        print("PARITY FAILED")
        sys.exit(1)
    print("Parity passed.")


if __name__ == "__main__":
    main()
