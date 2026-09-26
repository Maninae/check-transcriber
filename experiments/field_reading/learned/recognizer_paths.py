"""Where recognizer checkpoints and ONNX exports live (a leaf module: no project imports but config)."""

from experiments.field_reading.config import FIELD_READING_MODEL_ROOT

RECOGNIZER_ROOT = FIELD_READING_MODEL_ROOT / "recognizers"
ONNX_ROOT = RECOGNIZER_ROOT / "onnx"
CRNN_CHECKPOINTS = {
    "crnn_general": RECOGNIZER_ROOT / "crnn_general_h32" / "best.pt",
    "crnn_amount": RECOGNIZER_ROOT / "crnn_amount_h32" / "best.pt",
}
