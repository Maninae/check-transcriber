"""TrOCR-small's text codec, rebuilt on `sentencepiece` because transformers 5.x cannot load it.

The small TrOCR checkpoints ship only `sentencepiece.bpe.model` with `tokenizer_class:
XLMRobertaTokenizer`; transformers 5.17 has no slow tokenizer path left and its automatic
conversion fails (or maps everything to <unk>). This reproduces transformers 4.x's slow
XLMRobertaTokenizer id mapping exactly:
- fairseq specials <s>=0, <pad>=1, </s>=2, <unk>=3; every other piece id = spm id + 1;
- spm id 0 (spm's own unk) maps to 3; <mask> = spm vocab size + 1;
- encode wraps pieces as [<s>] + ids + [</s>]; decode drops specials and turns "▁" into spaces.
"""

from pathlib import Path

import sentencepiece

BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
FAIRSEQ_OFFSET = 1
SPECIAL_IDS = frozenset({BOS_ID, PAD_ID, EOS_ID, UNK_ID})
SENTENCEPIECE_FILE_NAME = "sentencepiece.bpe.model"


class XlmRobertaSentencePieceCodec:
    """Text <-> token ids for TrOCR-small, matching transformers 4.x XLMRobertaTokenizer."""

    pad_token_id = PAD_ID
    eos_token_id = EOS_ID
    bos_token_id = BOS_ID

    def __init__(self, sentencepiece_model_path: Path):
        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(sentencepiece_model_path))
        self.mask_token_id = self.processor.get_piece_size() + FAIRSEQ_OFFSET

    def encode(self, text: str) -> list[int]:
        """[<s>] + piece ids + [</s>]."""
        piece_ids = self.processor.encode(text, out_type=int)
        return [BOS_ID] + [piece_id + FAIRSEQ_OFFSET if piece_id != 0 else UNK_ID for piece_id in piece_ids] + [EOS_ID]

    def decode(self, token_ids: list[int]) -> str:
        """Text for model ids, skipping specials."""
        piece_ids = [token_id - FAIRSEQ_OFFSET for token_id in token_ids
                     if token_id not in SPECIAL_IDS and token_id < self.mask_token_id]
        return self.processor.decode(piece_ids).strip()

    def batch_decode(self, sequences: list[list[int]]) -> list[str]:
        """Decode several sequences."""
        return [self.decode(list(sequence)) for sequence in sequences]
