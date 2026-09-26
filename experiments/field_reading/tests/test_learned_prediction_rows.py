"""Tests for the reading-prediction row contract writer."""

import json

import pytest

from experiments.field_reading.learned.prediction_rows import (PREDICTION_ROW_KEYS, ReadingPredictionRow,
                                                               prediction_file_path, write_prediction_rows)


def make_row(**overrides) -> ReadingPredictionRow:
    values = dict(row_key="val_000001__check=0__field=date", scene_id="val_000001", check_index=0, field_name="date",
                  method="crnn_general", localization="oracle", pred_text="6/7/26", confidence=0.93,
                  pred_box=[1.0, 2.0, 30.0, 12.0], latency_ms=1.5)
    values.update(overrides)
    return ReadingPredictionRow(**values)


def test_rows_are_written_with_exact_contract_keys_in_order(tmp_path):
    output_path = tmp_path / "val" / "crnn_general__loc=oracle.jsonl"
    write_prediction_rows([make_row(), make_row(pred_text="", confidence=None, pred_box=None)], output_path)
    lines = output_path.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert tuple(json.loads(line).keys()) == PREDICTION_ROW_KEYS
    assert json.loads(lines[1])["pred_text"] == "" and json.loads(lines[1])["confidence"] is None


def test_confidence_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        make_row(confidence=1.2)


def test_prediction_file_path_layout():
    path = prediction_file_path("eval", "trocr_small_zeroshot", "layout_prior")
    assert path.parts[-2:] == ("eval", "trocr_small_zeroshot__loc=layout_prior.jsonl")
