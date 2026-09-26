"""Greedy TrOCR decoding with onnxruntime, for both ONNX layouts we ship or compare.

- `no_cache` (our onnx_export): decoder(input_ids (1, L), encoder_hidden_states) -> logits; the
  full prefix is re-run every step (O(L^2) but no cache plumbing).
- `merged_past` (Xenova/transformers.js `decoder_model_merged*.onnx`): 24 past tensors
  (6 layers x decoder/encoder x key/value, (1, 8, len, 32)) plus `use_cache_branch`; step 0 runs
  the no-cache branch on [</s>], later steps feed one token and the returned `present.*`.
Both start from </s> (id 2) like the PyTorch model and stop at </s> or TROCR_MAX_NEW_TOKENS.
`chosen_log_probabilities` returns each greedy token's log-softmax, so the ONNX confidence
exp(mean) matches TrocrCropReader's definition.
"""

import numpy as np
import onnxruntime

from experiments.field_reading.learned.trocr_reader import TROCR_MAX_NEW_TOKENS
from experiments.field_reading.learned.trocr_tokenizer import EOS_ID

PAST_INPUT_PREFIX = "past_key_values."
USE_CACHE_BRANCH_INPUT = "use_cache_branch"


def log_softmax_at(logits: np.ndarray, index: int) -> float:
    """log p(index) under softmax(logits), numerically stable."""
    shifted = logits - logits.max()
    return float(shifted[index] - np.log(np.exp(shifted).sum()))


def is_merged_past_decoder(decoder: onnxruntime.InferenceSession) -> bool:
    """True for the transformers.js merged decoder with past-KV inputs."""
    return any(item.name == USE_CACHE_BRANCH_INPUT for item in decoder.get_inputs())


def empty_past_inputs(decoder: onnxruntime.InferenceSession) -> dict[str, np.ndarray]:
    """Zero-length past tensors for the first (no-cache) step."""
    return {item.name: np.zeros((1, item.shape[1], 0, item.shape[3]), dtype=np.float32)
            for item in decoder.get_inputs() if item.name.startswith(PAST_INPUT_PREFIX)}


def greedy_decode_merged_past(decoder: onnxruntime.InferenceSession, hidden_states: np.ndarray) -> tuple[list[int], np.ndarray, list[float]]:
    """Greedy tokens, first-step logits and chosen-token log-probs using the KV cache."""
    output_names = [item.name for item in decoder.get_outputs()]
    past = empty_past_inputs(decoder)
    tokens, first_logits, chosen_log_probabilities = [EOS_ID], None, []
    for step in range(TROCR_MAX_NEW_TOKENS):
        feeds = {"input_ids": np.array([[tokens[-1]]], dtype=np.int64), "encoder_hidden_states": hidden_states,
                 USE_CACHE_BRANCH_INPUT: np.array([step > 0]), **past}
        outputs = dict(zip(output_names, decoder.run(None, feeds)))
        logits = outputs["logits"][0, -1]
        first_logits = logits if first_logits is None else first_logits
        tokens.append(int(logits.argmax()))
        chosen_log_probabilities.append(log_softmax_at(logits, tokens[-1]))
        for name, value in outputs.items():
            past_name = name.replace("present.", PAST_INPUT_PREFIX)
            # Cross-attention KV is computed at step 0; cache-branch steps return an empty (0, 8, 1, 32) placeholder.
            if name.startswith("present.") and value.size > 0:
                past[past_name] = value
        if tokens[-1] == EOS_ID:
            break
    return tokens, first_logits, chosen_log_probabilities


def greedy_decode_no_cache(decoder: onnxruntime.InferenceSession, hidden_states: np.ndarray) -> tuple[list[int], np.ndarray, list[float]]:
    """Greedy tokens, first-step logits and chosen-token log-probs, re-running the prefix each step."""
    tokens, first_logits, chosen_log_probabilities = [EOS_ID], None, []
    for _ in range(TROCR_MAX_NEW_TOKENS):
        logits = decoder.run(None, {"input_ids": np.array([tokens], dtype=np.int64), "encoder_hidden_states": hidden_states})[0][0, -1]
        first_logits = logits if first_logits is None else first_logits
        tokens.append(int(logits.argmax()))
        chosen_log_probabilities.append(log_softmax_at(logits, tokens[-1]))
        if tokens[-1] == EOS_ID:
            break
    return tokens, first_logits, chosen_log_probabilities


def onnx_trocr_greedy(encoder: onnxruntime.InferenceSession, decoder: onnxruntime.InferenceSession,
                      pixel_values: np.ndarray) -> tuple[list[int], np.ndarray, np.ndarray, float]:
    """Encode one image and greedy-decode it; returns tokens, encoder states, first-step logits, confidence."""
    hidden_states = encoder.run(None, {encoder.get_inputs()[0].name: pixel_values})[0]
    decode = greedy_decode_merged_past if is_merged_past_decoder(decoder) else greedy_decode_no_cache
    tokens, first_logits, chosen_log_probabilities = decode(decoder, hidden_states)
    return tokens, hidden_states, first_logits, float(np.exp(np.mean(chosen_log_probabilities)))
