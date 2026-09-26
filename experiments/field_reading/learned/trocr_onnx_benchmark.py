"""Equivalence, size and CPU latency of TrOCR ONNX graph pairs vs the PyTorch model.

Variants compared (all TrOCR-small-handwritten, grey+autocontrast input as shipped):
- ours_no_cache_{fp32,int8}: our onnx_export (decoder re-runs the prefix each step);
- xenova_past_{fp32,int8}: Xenova/trocr-small-handwritten (transformers.js) encoder +
  decoder_model_merged with KV cache; the files the browser would actually load.
Per variant: identical greedy text on EQUIVALENCE_CROP_COUNT val crops, max abs diff of encoder
states and first-step logits, total MB, and median full-greedy latency per crop over
LATENCY_CROP_COUNT crops with 1 intra-op thread and onnxruntime's default.

Run: python -m experiments.field_reading.learned.trocr_onnx_benchmark
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from experiments.field_reading.learned.onnx_benchmark import (EQUIVALENCE_CROP_COUNT, LATENCY_CROP_COUNT, median_latency_ms,
                                                              merge_into_benchmark_report, ort_session, sample_val_crops)
from experiments.field_reading.learned.onnx_export import ONNX_ROOT
from experiments.field_reading.learned.trocr_onnx_decoding import onnx_trocr_greedy
from experiments.field_reading.learned.trocr_reader import TrocrCropReader, trocr_pixel_values
from experiments.field_reading.learned.trocr_tokenizer import EOS_ID

logger = logging.getLogger(__name__)

MODEL_ID = "trocr_small_handwritten"
XENOVA_REPOSITORY = "Xenova/trocr-small-handwritten"


def variant_paths() -> dict[str, tuple[Path, Path]]:
    """label -> (encoder, decoder) ONNX files."""
    xenova = Path(snapshot_download(XENOVA_REPOSITORY, allow_patterns=["*.json", "onnx/encoder_model*.onnx",
                                                                     "onnx/decoder_model_merged*.onnx"])) / "onnx"
    return {
        "ours_no_cache_fp32": (ONNX_ROOT / f"{MODEL_ID}.encoder.onnx", ONNX_ROOT / f"{MODEL_ID}.decoder.onnx"),
        "ours_no_cache_int8": (ONNX_ROOT / f"{MODEL_ID}.encoder.int8.onnx", ONNX_ROOT / f"{MODEL_ID}.decoder.int8.onnx"),
        "xenova_past_fp32": (xenova / "encoder_model.onnx", xenova / "decoder_model_merged.onnx"),
        "xenova_past_int8": (xenova / "encoder_model_quantized.onnx", xenova / "decoder_model_merged_quantized.onnx"),
    }


def main() -> None:
    """Benchmark every variant, print and merge into reports/learned/onnx_benchmark.json."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(2)
    reader = TrocrCropReader(MODEL_ID, "cpu", grey_input=True)
    _, crops = sample_val_crops(LATENCY_CROP_COUNT, None)
    pixel_batches = [trocr_pixel_values([crop], grey_input=True).numpy() for crop in crops]
    equivalence_pixels = pixel_batches[:EQUIVALENCE_CROP_COUNT]
    torch_texts = [text for text, _ in reader.read_crops(crops[:EQUIVALENCE_CROP_COUNT])]
    with torch.no_grad():
        torch_hidden = [reader.model.encoder(pixel_values=torch.from_numpy(p)).last_hidden_state.numpy() for p in equivalence_pixels]
        torch_first_logits = [reader.model.decoder(input_ids=torch.tensor([[EOS_ID]]), encoder_hidden_states=torch.from_numpy(h)).logits[0, -1].numpy()
                              for h in torch_hidden]
    report = {"model": "trocr_small_handwritten_onnx", "pytorch_reference": "fp32 CPU greedy, grey+autocontrast input"}
    for label, (encoder_path, decoder_path) in variant_paths().items():
        if not encoder_path.exists():
            logger.warning("skipping %s: %s missing", label, encoder_path)
            continue
        encoder, decoder = ort_session(encoder_path, None), ort_session(decoder_path, None)
        identical, encoder_difference, logit_difference = 0, 0.0, 0.0
        for pixels, text, hidden, first_logits in zip(equivalence_pixels, torch_texts, torch_hidden, torch_first_logits):
            tokens, onnx_hidden, onnx_first_logits = onnx_trocr_greedy(encoder, decoder, pixels)
            identical += reader.codec.decode(tokens) == text
            encoder_difference = max(encoder_difference, float(np.abs(onnx_hidden - hidden).max()))
            logit_difference = max(logit_difference, float(np.abs(onnx_first_logits - first_logits).max()))
        entry = {"size_mb": round((encoder_path.stat().st_size + decoder_path.stat().st_size) / 1e6, 1),
                 "identical_text": f"{identical}/{EQUIVALENCE_CROP_COUNT}",
                 "max_abs_diff_encoder_states": round(encoder_difference, 5),
                 "max_abs_diff_first_step_logits": round(logit_difference, 5)}
        for threads, key in [(1, "latency_ms_per_crop_1thread"), (None, "latency_ms_per_crop_default_threads")]:
            timed_encoder, timed_decoder = ort_session(encoder_path, threads), ort_session(decoder_path, threads)
            entry[key] = round(median_latency_ms(lambda pixels: onnx_trocr_greedy(timed_encoder, timed_decoder, pixels), pixel_batches), 1)
        report[label] = entry
        logger.info("%s: %s", label, entry)
    merge_into_benchmark_report([report])
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
