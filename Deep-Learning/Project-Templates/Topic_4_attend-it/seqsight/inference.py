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


# ----------------------------------------------------------------------------- guardrails
# A sentiment score is only worth acting on when the model is in familiar
# territory. Below these limits the API still returns the model's answer (for
# transparency) but marks the prediction as abstained and the verdict falls back
# to the base rate: IMDB is 50/50, so "uncertain" rather than a coin-flip label.
LOW_CONFIDENCE = 0.75   # test set: answering only at >= 0.75 is far more accurate (models/runs.json "selective")
MIN_TOKENS = 5          # too short to carry much sentiment signal
MAX_UNK_SHARE = 0.2     # over a fifth of the words were never seen in training
FLAG_TEXT = {
    "low_confidence": f"confidence below {LOW_CONFIDENCE:.0%}",
    "too_short": f"fewer than {MIN_TOKENS} words",
    "many_unknown_words": f"more than {MAX_UNK_SHARE:.0%} of words are outside the vocabulary",
    "truncated": "only the first 256 tokens were read",
}
ABSTAIN_FLAGS = {"low_confidence", "too_short", "many_unknown_words"}


def guardrails(res: Dict) -> Dict:
    """Flags for a classify_tokens result, whether to abstain, and the verdict to show."""
    flags = []
    if res["confidence"] < LOW_CONFIDENCE:
        flags.append("low_confidence")
    if res["n_tokens"] < MIN_TOKENS:
        flags.append("too_short")
    if res["unk_tokens"] > MAX_UNK_SHARE * res["n_tokens"]:
        flags.append("many_unknown_words")
    if res["truncated"]:
        flags.append("truncated")
    abstained = bool(ABSTAIN_FLAGS.intersection(flags))
    return {"flags": flags, "flag_messages": [FLAG_TEXT[f] for f in flags], "abstained": abstained,
            "verdict": "uncertain" if abstained else res["class_name"]}


def top_tokens(tokens: Sequence[str], attention: Sequence[float], k: int = 5) -> List[Dict]:
    """The k most-attended tokens, highest first (ties keep reading order)."""
    order = sorted(range(len(tokens)), key=lambda i: -attention[i])[:k]
    return [{"position": i, "token": tokens[i], "weight": float(attention[i])} for i in order]
