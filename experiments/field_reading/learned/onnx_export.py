"""Export the recognizers to ONNX (fp32 + dynamic int8) for onnxruntime-web.

Outputs under ONNX_ROOT = RECOGNIZER_ROOT / "onnx":
- `<crnn run>.onnx` / `<crnn run>.int8.onnx`: input `line_images` (B, 1, H, W) float in [-1, 1]
  (recipe: line_image_preprocessing), output `log_probabilities` (B, W // 4, C). B and W dynamic.
  A sidecar `<crnn run>.json` carries the charset, input height and width limits for the app.
- `style_classifier_h32[.int8].onnx`: (line_images (B, 1, 32, W), valid_width_fraction (B,)) ->
  handwritten logit (B,); same input recipe as the CRNN.
- TrOCR as two graphs, no KV cache (greedy loop re-runs the decoder, fine for <= 48 tokens):
  `<id>.encoder.onnx`: pixel_values (B, 3, 384, 384) -> encoder_hidden_states;
  `<id>.decoder.onnx`: (input_ids (B, L), encoder_hidden_states) -> logits (B, L, V).

Run: python -m experiments.field_reading.learned.onnx_export --crnn crnn_general_h32 crnn_amount_h32 --style [--trocr trocr_small_handwritten]
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from experiments.field_reading.learned.crnn_reader import CrnnCropReader
from experiments.field_reading.learned.handwriting_style_classifier import STYLE_CHECKPOINT_PATH, HandwritingStyleClassifier
from experiments.field_reading.learned.recognizer_paths import ONNX_ROOT, RECOGNIZER_ROOT
from experiments.field_reading.learned.trocr_reader import TROCR_INPUT_SIZE, TrocrCropReader

logger = logging.getLogger(__name__)

ONNX_OPSET = 17
EXAMPLE_WIDTH = 256


def quantize_to_int8(fp32_path: Path) -> Path:
    """Dynamic (weight-only) int8 quantization next to the fp32 file."""
    int8_path = fp32_path.with_suffix(".int8.onnx")
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    return int8_path


def export_crnn(run_name: str) -> list[Path]:
    """CRNN checkpoint -> fp32 + int8 ONNX and the app sidecar JSON."""
    reader = CrnnCropReader(RECOGNIZER_ROOT / run_name / "best.pt", "cpu")
    example = torch.zeros(1, 1, reader.variant.input_height, EXAMPLE_WIDTH)
    fp32_path = ONNX_ROOT / f"{run_name}.onnx"
    torch.onnx.export(reader.model, (example,), str(fp32_path), input_names=["line_images"], output_names=["log_probabilities"],
                      dynamic_axes={"line_images": {0: "batch", 3: "width"}, "log_probabilities": {0: "batch", 1: "timesteps"}},
                      opset_version=ONNX_OPSET, dynamo=False)
    sidecar = {"charset": reader.charset.characters, "blank_index": 0, "input_height": reader.variant.input_height,
               "min_width": reader.variant.min_width, "max_width": reader.variant.max_width, "width_downsample": 4,
               "preprocessing": "luma grey, bilinear resize to input_height keeping aspect, clamp width, (x/255-0.5)/0.5, right-pad 1.0"}
    (ONNX_ROOT / f"{run_name}.json").write_text(json.dumps(sidecar, indent=1))
    return [fp32_path, quantize_to_int8(fp32_path)]


def export_style_classifier() -> list[Path]:
    """Handwriting style classifier -> fp32 + int8 ONNX."""
    model = HandwritingStyleClassifier().eval()
    model.load_state_dict(torch.load(STYLE_CHECKPOINT_PATH, map_location="cpu"))
    fp32_path = ONNX_ROOT / "style_classifier_h32.onnx"
    torch.onnx.export(model, (torch.zeros(1, 1, 32, EXAMPLE_WIDTH), torch.ones(1)), str(fp32_path),
                      input_names=["line_images", "valid_width_fraction"], output_names=["handwritten_logit"],
                      dynamic_axes={"line_images": {0: "batch", 3: "width"}, "valid_width_fraction": {0: "batch"},
                                    "handwritten_logit": {0: "batch"}}, opset_version=ONNX_OPSET, dynamo=False)
    return [fp32_path, quantize_to_int8(fp32_path)]


class TrocrDecoderStep(nn.Module):
    """Decoder + LM head as one graph: (input_ids, encoder_hidden_states) -> logits."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=False).logits


class TrocrEncoder(nn.Module):
    """Vision encoder as one graph: pixel_values -> last hidden states."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values=pixel_values).last_hidden_state


def export_trocr(model_id: str) -> list[Path]:
    """TrOCR -> encoder/decoder fp32 + int8 ONNX files."""
    model = TrocrCropReader(model_id, "cpu").model.eval()
    pixel_values = torch.zeros(1, 3, TROCR_INPUT_SIZE, TROCR_INPUT_SIZE)
    encoder_path, decoder_path = ONNX_ROOT / f"{model_id}.encoder.onnx", ONNX_ROOT / f"{model_id}.decoder.onnx"
    encoder = TrocrEncoder(model.encoder)
    torch.onnx.export(encoder, (pixel_values,), str(encoder_path), input_names=["pixel_values"],
                      output_names=["encoder_hidden_states"], dynamic_axes={"pixel_values": {0: "batch"},
                                                                          "encoder_hidden_states": {0: "batch"}},
                      opset_version=ONNX_OPSET, dynamo=False)
    with torch.no_grad():
        hidden_states = encoder(pixel_values)
    input_ids = torch.tensor([[2, 100, 200]])
    torch.onnx.export(TrocrDecoderStep(model.decoder), (input_ids, hidden_states), str(decoder_path),
                      input_names=["input_ids", "encoder_hidden_states"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "length"}, "encoder_hidden_states": {0: "batch"},
                                    "logits": {0: "batch", 1: "length"}}, opset_version=ONNX_OPSET, dynamo=False)
    return [encoder_path, decoder_path, quantize_to_int8(encoder_path), quantize_to_int8(decoder_path)]


def main() -> None:
    """Export the requested models."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crnn", nargs="*", default=[])
    parser.add_argument("--trocr", nargs="*", default=[])
    parser.add_argument("--style", action="store_true", help="also export the handwriting style classifier")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ONNX_ROOT.mkdir(parents=True, exist_ok=True)
    written = [path for run in arguments.crnn for path in export_crnn(run)]
    written += [path for model_id in arguments.trocr for path in export_trocr(model_id)]
    written += export_style_classifier() if arguments.style else []
    for path in written:
        logger.info("%s %.2f MB", path.name, path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
