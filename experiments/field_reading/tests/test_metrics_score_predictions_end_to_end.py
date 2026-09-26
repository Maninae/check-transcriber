"""End-to-end: the score_predictions CLI on ~50 real val rows writes a complete JSON + markdown report."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from experiments.field_reading.config import SYNTH_V1_ROOT
from experiments.field_reading.data_access.field_manifest import load_field_rows
from experiments.field_reading.metrics.hard_set import build_hard_set_flags, write_hard_set_flags

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
TOY_ROW_COUNT = 56

pytestmark = pytest.mark.skipif(not (SYNTH_V1_ROOT / "ocr").exists(), reason="synth v1 on vega not mounted")


def write_toy_predictions(toy_rows, prediction_path: Path) -> None:
    """GT text on printed rows (confidence 0.9), blank on handwritten; one extra row for a missing-prediction case."""
    with prediction_path.open("w") as prediction_file:
        for row in toy_rows.iloc[1:].itertuples():  # row 0 left out: a missing prediction counts as blank
            prediction_file.write(json.dumps({
                "row_key": row.row_key, "scene_id": row.scene_id, "check_index": row.check_index,
                "field_name": row.field_name, "method": "toy", "localization": "oracle",
                "pred_text": "" if row.handwritten else row.text, "confidence": 0.9, "pred_box": None, "latency_ms": 1.0,
            }) + "\n")


def test_cli_writes_json_and_markdown_with_every_section(tmp_path: Path) -> None:
    """Run the CLI as a user would; check counts, sections and the known accuracy of the toy method."""
    field_rows = load_field_rows("val")
    scored_status_rows = field_rows[field_rows.status.isin(["ok", "too_small"])]
    toy_rows = pd.concat([scored_status_rows[scored_status_rows.status.eq(status)].sample(TOY_ROW_COUNT // 2, random_state=0)
                          for status in ["ok", "too_small"]])
    flags_path = tmp_path / "flags.jsonl"
    write_hard_set_flags(build_hard_set_flags(toy_rows, worker_count=1), flags_path)
    prediction_path = tmp_path / "toy__loc=oracle.jsonl"
    write_toy_predictions(toy_rows, prediction_path)
    subprocess.run([sys.executable, "-m", "experiments.field_reading.metrics.score_predictions", "--split", "val",
                    "--predictions", str(prediction_path), "--only-predicted-rows", "--hard-set-flags", str(flags_path),
                    "--output-dir", str(tmp_path), "--run-name", "toy_run"], cwd=WORKTREE_ROOT, check=True)
    report = json.loads((tmp_path / "toy_run.json").read_text())
    markdown_text = (tmp_path / "toy_run.md").read_text()
    predicted_rows = toy_rows.iloc[1:]
    ok_rows = predicted_rows[predicted_rows.status.eq("ok")]
    pooled_headline = next(record for record in report["headline"] if record["field_name"] == "ALL")
    assert pooled_headline["n"] == len(ok_rows)
    assert pooled_headline["accuracy"] == pytest.approx((~ok_rows.handwritten.astype(bool)).mean())
    assert pooled_headline["accuracy_on_filled"] == pytest.approx(1.0)
    assert {"headline", "gating", "hard_set", "hard_set_by_flag", "too_small", "status", "breakdowns"} <= report.keys()
    assert {"handwritten", "pen_font_id", "text_height_bin", "layout_family"} <= report["breakdowns"].keys()
    for heading in ["## Headline", "## Confidence gating", "## Hard set", "## too_small rows", "### Pen font", "### Layout family"]:
        assert heading in markdown_text
