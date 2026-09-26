"""Learned text recognizers (U4). See CLAUDE.md for the module map.

Model caches must live on vega, and huggingface_hub fixes its cache path when it is first
imported, so the defaults are set here: this package init runs before any submodule imports
transformers / huggingface_hub.
"""

import os

from experiments.field_reading.config import FIELD_READING_MODEL_ROOT

os.environ.setdefault("HF_HOME", str(FIELD_READING_MODEL_ROOT / "hf_home"))
os.environ.setdefault("TORCH_HOME", str(FIELD_READING_MODEL_ROOT / "torch_home"))
