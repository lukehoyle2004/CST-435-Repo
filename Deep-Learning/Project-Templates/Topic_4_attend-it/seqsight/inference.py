"""Turn review text into a prediction plus per-token attention, for the API tier.

The same tokenizer and vocab the models were trained with (``shared.text`` and
``models/vocab.json``) are used here, so a review is encoded exactly as it was
in training: normalize -> tokenize -> keep the first MAX_LEN tokens -> ids.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

from torch import nn

from seqsight import MODELS_DIR
from seqsight.model import predict_ids
from shared.imdb import CLASS_NAMES
from shared.text import MAX_LEN, UNK_ID, Vocab, tokenize


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_vocab(models_dir: Path = MODELS_DIR) -> Vocab:
    return Vocab.from_dict(json.loads((models_dir / "vocab.json").read_text()))


def classify_tokens(model: nn.Module, vocab: Vocab, tokens: Sequence[str],
                    raw_tokens: int | None = None) -> Dict:
    """Score already-tokenized text. Raises ValueError if there is nothing to score.

    ``raw_tokens`` is the length before truncation when ``tokens`` was already cut
    to MAX_LEN (the gallery stores only the kept tokens).
    """
    kept = list(tokens[:MAX_LEN])
    if not kept:
        raise ValueError("no scorable tokens (letters, digits, ! or ?)")
    raw = len(tokens) if raw_tokens is None else raw_tokens
    ids = vocab.encode(kept)
    probs, weights = predict_ids(model, ids)
    label = max(range(len(probs)), key=probs.__getitem__)
    return dict(
        label=label,
        class_name=CLASS_NAMES[label],
        confidence=float(probs[label]),
        probabilities={c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        tokens=kept,
        unk=[i == UNK_ID for i in ids],
        attention=[round(w, 6) for w in weights] if weights is not None else None,
        n_tokens=len(kept),
        raw_tokens=raw,
        truncated=raw > len(kept),
        unk_tokens=sum(i == UNK_ID for i in ids),
    )


def classify_text(model: nn.Module, vocab: Vocab, text: str) -> Dict:
    return classify_tokens(model, vocab, tokenize(text))


def top_tokens(tokens: Sequence[str], attention: Sequence[float], k: int = 5) -> List[Dict]:
    """The k most-attended tokens, highest first (ties keep reading order)."""
    order = sorted(range(len(tokens)), key=lambda i: -attention[i])[:k]
    return [{"position": i, "token": tokens[i], "weight": float(attention[i])} for i in order]
