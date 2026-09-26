"""The browser's field reader, step for step, in Python (the reference the browser is compared to).

Mirrors app/js/pipeline/fields/ on the same RGB pixels, reusing experiments/field_reading's own
functions wherever they exist so the reference IS the research code:
- canvas: half-size INTER_AREA (stands in for the half-resolution JPEG decode segnet trained on)
  + segnet_dataset's crop->canvas homography; `segnet_postprocess.logits_to_field_boxes`; the
  val-selected box gate from `segnet_mobilenetv3l_768__gate.json`;
- `field_crop.crop_field_from_check`, `line_image_preprocessing.preprocess_line_image`'s recipe with
  INTER_LINEAR_EXACT (the browser's resize; `research_line_resize=True` uses the research's own
  INTER_LINEAR, which on an ARM Mac is a vendor HAL no browser reproduces), `ctc_decoding.decode_ctc_batch`
  (mean_char);
- each field read alone but right-padded to a multiple of 128 px (as trained), valid timesteps decoded.
- optional TrOCR: Xenova fp32 encoder + int8 merged decoder via `trocr_onnx_decoding` with the
  grey+autocontrast recipe of `trocr_reader.trocr_pixel_values`.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from experiments.field_reading.data_access.field_crop import crop_field_from_check
from experiments.field_reading.field_localization.segnet_postprocess import MASK_THRESHOLD, logits_to_field_boxes
from experiments.field_reading.learned.ctc_decoding import CtcConfidenceKind, decode_ctc_batch
from experiments.field_reading.learned.line_image_preprocessing import preprocess_line_image
from experiments.field_reading.learned.text_charset import CHARSET_REGISTRY

APP_MODELS_DIRECTORY = Path(__file__).resolve().parents[2] / "models"
SEGNET_GATE_PATH = Path("/Volumes/vega/ai-models/field-reading/localization/segnet_mobilenetv3l_768__gate.json")
FIELD_NAMES = ["payer_name", "payee", "amount_numeric", "amount_words", "date", "memo", "check_number"]
PRINTED_READER_BY_FIELD = {name: "crnn_general" for name in FIELD_NAMES} | {"amount_numeric": "crnn_amount"}
CANVAS_WIDTH, CANVAS_HEIGHT = 768, 352
LINE_HEIGHT, LINE_MIN_WIDTH, LINE_MAX_WIDTH = 32, 48, 800
STYLE_PAD_MULTIPLE = 128
HANDWRITTEN_THRESHOLD = 0.5
MIN_CROP_SIDE_PX = 2
SESSION_THREADS = 2  # shared machine


def open_session(path: Path) -> onnxruntime.InferenceSession:
    """CPU onnxruntime session with a small thread budget."""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = SESSION_THREADS
    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def build_segnet_canvas(crop_rgb: np.ndarray) -> np.ndarray:
    """(1, 3, 352, 768) float32 in [0, 1]: the browser's canvas recipe."""
    crop_height, crop_width = crop_rgb.shape[:2]
    reduced_width, reduced_height = math.ceil(crop_width / 2), math.ceil(crop_height / 2)
    reduced = cv2.resize(crop_rgb, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA)
    crop_to_canvas = np.diag([CANVAS_WIDTH / crop_width, CANVAS_WIDTH / crop_width, 1.0])
    reduced_to_crop = np.diag([crop_width / reduced_width, crop_height / reduced_height, 1.0])
    canvas = cv2.warpPerspective(reduced, crop_to_canvas @ reduced_to_crop, (CANVAS_WIDTH, CANVAS_HEIGHT),
                                 flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None].copy()


def pad_line(line_image: np.ndarray) -> np.ndarray:
    """(1, 1, 32, W') batch of one line right-padded with 1.0 to a multiple of 128 px."""
    width = line_image.shape[1]
    padded_width = -(-width // STYLE_PAD_MULTIPLE) * STYLE_PAD_MULTIPLE
    padded = np.full((1, 1, LINE_HEIGHT, padded_width), 1.0, dtype=np.float32)
    padded[0, 0, :, :width] = line_image
    return padded


def style_classifier_inputs(line_image: np.ndarray) -> dict[str, np.ndarray]:
    """The padded line plus its valid width fraction."""
    padded = pad_line(line_image)
    return {"line_images": padded, "valid_width_fraction": np.array([line_image.shape[1] / padded.shape[-1]], dtype=np.float32)}


def browser_line_image(field_crop_rgb: np.ndarray) -> np.ndarray:
    """`preprocess_line_image` with INTER_LINEAR_EXACT (see module docstring)."""
    grey = cv2.cvtColor(field_crop_rgb, cv2.COLOR_RGB2GRAY)
    height, width = grey.shape
    target_width = int(np.clip(int(round(width * LINE_HEIGHT / max(1, height))), LINE_MIN_WIDTH, LINE_MAX_WIDTH))
    resized = cv2.resize(grey, (target_width, LINE_HEIGHT), interpolation=cv2.INTER_LINEAR_EXACT)
    return (resized.astype(np.float32) / 255.0 - 0.5) / 0.5


class PythonFieldReader:
    """Reads a rectified RGB crop into the browser's rawReads dict (see module docstring)."""

    def __init__(self, handwriting_reader=None, research_line_resize: bool = False):
        self.research_line_resize = research_line_resize
        self.segnet = open_session(APP_MODELS_DIRECTORY / "segnet_mobilenetv3l_768.onnx")
        self.crnn = {"crnn_general": open_session(APP_MODELS_DIRECTORY / "crnn_general_h32.onnx"),
                     "crnn_amount": open_session(APP_MODELS_DIRECTORY / "crnn_amount_h32.onnx")}
        self.charsets = {"crnn_general": CHARSET_REGISTRY["general"], "crnn_amount": CHARSET_REGISTRY["amount"]}
        self.style = open_session(APP_MODELS_DIRECTORY / "style_classifier_h32.onnx")
        self.box_gate = json.loads(SEGNET_GATE_PATH.read_text())["min_confidence"]
        self.handwriting_reader = handwriting_reader

    def locate(self, crop_rgb: np.ndarray) -> dict[str, list[float] | None]:
        """Field name -> gated box (or None)."""
        crop_height, crop_width = crop_rgb.shape[:2]
        logits = self.segnet.run(None, {"canvas_rgb": build_segnet_canvas(crop_rgb)})[0][0]
        located = logits_to_field_boxes(logits, CANVAS_WIDTH / crop_width, MASK_THRESHOLD, (crop_width, crop_height))
        return {name: (box if box is not None and confidence >= self.box_gate else None)
                for name, (box, confidence) in zip(FIELD_NAMES, located)}

    def read_crnn(self, reader_id: str, line_image: np.ndarray) -> tuple[str, float]:
        """(text, mean_char confidence) for one line."""
        log_probabilities = self.crnn[reader_id].run(None, {"line_images": pad_line(line_image)})[0]
        [(text, confidences)] = decode_ctc_batch(log_probabilities, [line_image.shape[1] // 4], self.charsets[reader_id],
                                                 CtcConfidenceKind.MEAN_CHAR)
        return text, confidences["confidence"]

    def read(self, crop_rgb: np.ndarray) -> dict[str, dict | None]:
        """rawReads for one crop, in the browser's shape."""
        raw_reads = {}
        for field_name, box in self.locate(crop_rgb).items():
            if box is None:
                raw_reads[field_name] = None
                continue
            field_crop = crop_field_from_check(crop_rgb, box)
            if min(field_crop.shape[:2]) < MIN_CROP_SIDE_PX:
                raw_reads[field_name] = None
                continue
            line_image = (preprocess_line_image(field_crop, LINE_HEIGHT, LINE_MIN_WIDTH, LINE_MAX_WIDTH)
                          if self.research_line_resize else browser_line_image(field_crop))
            logit = float(self.style.run(None, style_classifier_inputs(line_image))[0][0])
            handwritten_probability = 1 / (1 + math.exp(-logit))
            reader_id = PRINTED_READER_BY_FIELD[field_name]
            use_handwriting = self.handwriting_reader is not None and handwritten_probability > HANDWRITTEN_THRESHOLD
            if not use_handwriting:
                text, confidence = self.read_crnn(reader_id, line_image)
            else:
                text, confidence = self.handwriting_reader.read(field_crop)
                reader_id = "trocr"
                if field_name == "amount_numeric":
                    printed_text, printed_confidence = self.read_crnn("crnn_amount", line_image)
                    if not confidence > printed_confidence:
                        text, confidence, reader_id = printed_text, printed_confidence, "crnn_amount"
            raw_reads[field_name] = {"text": text, "confidence": confidence, "handwrittenProbability": handwritten_probability,
                                     "reader": reader_id, "box": [float(value) for value in box]}
        return raw_reads


class PythonHandwritingReader:
    """Xenova TrOCR (fp32 encoder + int8 merged decoder), grey + autocontrast input."""

    def __init__(self):
        from experiments.field_reading.learned.trocr_onnx_benchmark import variant_paths
        from experiments.field_reading.learned.trocr_reader import resolve_trocr_directory
        from experiments.field_reading.learned.trocr_tokenizer import SENTENCEPIECE_FILE_NAME, XlmRobertaSentencePieceCodec
        encoder_path, decoder_path = variant_paths()["xenova_past_fp32enc_int8dec"]
        self.encoder, self.decoder = open_session(encoder_path), open_session(decoder_path)
        self.codec = XlmRobertaSentencePieceCodec(resolve_trocr_directory("trocr_small_handwritten") / SENTENCEPIECE_FILE_NAME)

    def read(self, field_crop_rgb: np.ndarray) -> tuple[str, float]:
        """(text, sequence confidence)."""
        from experiments.field_reading.learned.trocr_onnx_decoding import onnx_trocr_greedy
        from experiments.field_reading.learned.trocr_reader import trocr_pixel_values
        tokens, _, _, confidence = onnx_trocr_greedy(self.encoder, self.decoder, trocr_pixel_values([field_crop_rgb], True).numpy())
        return self.codec.decode(tokens), confidence
