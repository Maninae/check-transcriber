"""Read one field crop with Tesseract (tesserocr), returning text and a confidence in [0, 1].

Engine parity with the app: tesserocr 5.5.1 (libtesseract 5.5) with the same `eng`
traineddata the app's Tesseract.js 7 loads (`@tesseract.js-data/eng` 4.0.0_best_int), LSTM only.

Confidence is Tesseract's mean word confidence / 100 (MeanTextConf); empty output -> 0.
One `PyTessBaseAPI` per (process, psm, whitelist), cached, because creating one costs ~100 ms.
"""

import logging
from dataclasses import dataclass

import numpy as np
import tesserocr
from PIL import Image

from experiments.field_reading.config import TESSDATA_ROOT
from experiments.field_reading.ocr_baselines.field_text_cleanup import clean_field_text
from experiments.field_reading.ocr_baselines.field_crop_preprocessing import (TesseractPreprocessingConfig,
                                                                                preprocess_field_crop)

logger = logging.getLogger(__name__)

TESSERACT_LANGUAGE = "eng"
TESSERACT_DPI = "300"   # silences "Invalid resolution" and fixes the internal size heuristics

# Character whitelists per field; None = unrestricted. Letters in DATE cover month names/abbreviations.
FIELD_CHARACTER_WHITELISTS: dict[str, str] = {
    "amount_numeric": "0123456789$*,.-/=xX",
    "check_number": "0123456789",
    "date": "0123456789/-., ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "amount_words": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-&/*,. ",
}


@dataclass(frozen=True)
class TesseractReadConfig:
    """Everything that determines one Tesseract read of a field crop."""

    page_segmentation_mode: int = 7   # 7 single line, 13 raw line, 6 uniform block
    use_field_whitelist: bool = False
    preprocessing: TesseractPreprocessingConfig = TesseractPreprocessingConfig()

    def config_id(self) -> str:
        """Short stable id used in method names and logs."""
        p = self.preprocessing
        return (f"psm{self.page_segmentation_mode}__h={p.target_crop_height_px or 'native'}__bin={p.binarization}"
                f"__rule_removal={'on' if p.remove_ruled_lines else 'off'}__wl={'on' if self.use_field_whitelist else 'off'}")


TESSERACT_API_CACHE: dict[tuple[int, str], tesserocr.PyTessBaseAPI] = {}


def tesseract_api_for(page_segmentation_mode: int, whitelist: str) -> tesserocr.PyTessBaseAPI:
    """Process-local cached API configured for one psm + whitelist."""
    cache_key = (page_segmentation_mode, whitelist)
    if cache_key not in TESSERACT_API_CACHE:
        api = tesserocr.PyTessBaseAPI(path=str(TESSDATA_ROOT), lang=TESSERACT_LANGUAGE,
                                      psm=page_segmentation_mode, oem=tesserocr.OEM.LSTM_ONLY)
        api.SetVariable("user_defined_dpi", TESSERACT_DPI)
        if whitelist:
            api.SetVariable("tessedit_char_whitelist", whitelist)
        TESSERACT_API_CACHE[cache_key] = api
    return TESSERACT_API_CACHE[cache_key]


def read_field_crop_with_tesseract(crop_rgb: np.ndarray, field_name: str, config: TesseractReadConfig) -> tuple[str, float]:
    """(cleaned text, confidence in [0, 1]) for one field crop; see field_text_cleanup for the rules."""
    whitelist = FIELD_CHARACTER_WHITELISTS.get(field_name, "") if config.use_field_whitelist else ""
    api = tesseract_api_for(config.page_segmentation_mode, whitelist)
    api.SetImage(Image.fromarray(preprocess_field_crop(crop_rgb, config.preprocessing)))
    text = clean_field_text(field_name, " ".join(api.GetUTF8Text().split()))
    confidence = api.MeanTextConf() / 100.0 if text else 0.0
    return text, max(0.0, min(1.0, confidence))
