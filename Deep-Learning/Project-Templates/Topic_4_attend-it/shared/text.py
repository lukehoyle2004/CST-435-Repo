"""Tokenization, vocabulary, padding and masks for IMDB reviews.

Lives in ``shared`` so the training script, the API and the tests all turn text
into token ids the *same* way. Numpy only (no torch), so it can be imported and
unit-tested anywhere.

Pipeline (text -> model input)
------------------------------
1. ``normalize``  lowercase, turn IMDB's ``<br />`` tags into spaces, unescape
   HTML entities (``&amp;`` -> ``&``).
2. ``tokenize``   word-level regex split. A token is a run of letters/digits,
   optionally followed by an apostrophe suffix, so contractions stay whole
   (``don't``, ``isn't``, ``film's``): negation matters for sentiment and we
   want the attention bar to land on ``isn't``, not on ``isn`` + ``t``. ``!``
   and ``?`` are kept as their own tokens because they carry tone; all other
   punctuation is dropped.
3. ``Vocab``      maps tokens to ids. Built from the *training* split only.
   Id 0 is ``<pad>``, id 1 is ``<unk>`` (any word not in the vocab). Real words
   follow, most frequent first, ties broken alphabetically so the same corpus
   always yields the same ids. ``min_freq`` drops typos and one-off names;
   ``max_size`` caps the embedding table.
4. ``encode``     tokens -> ids, truncated to ``max_len`` (keeps the *first*
   ``max_len`` tokens).
5. ``pad_batch``  right-pads a batch of id lists with 0 to the longest length
   in the batch and returns the true ``lengths``. Right padding is what
   ``torch.nn.utils.rnn.pack_padded_sequence`` expects.
6. ``padding_mask`` boolean ``[B, T]``, True on real tokens, False on padding.
   The attention block sets the scores at False positions to -inf before the
   softmax so padding gets exactly zero weight.

Why word-level and not a pretrained subword tokenizer: the LSTM is trained from
scratch, so there are no pretrained embeddings to stay aligned with, and whole
words make the per-token attention readable for a non-technical client.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
SPECIALS: List[str] = [PAD_TOKEN, UNK_TOKEN]

MAX_LEN = 256          # tokens kept per review
MAX_VOCAB = 20_000     # including the two specials
MIN_FREQ = 2           # a word must appear this often in training to get an id

_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")


def normalize(text: str) -> str:
    """Lowercase, replace ``<br />`` with a space, unescape HTML entities."""
    return html.unescape(_BR.sub(" ", text)).lower()


def tokenize(text: str) -> List[str]:
    """Split a raw review into word-level tokens (see module docstring)."""
    return _TOKEN.findall(normalize(text).replace("’", "'"))


class Vocab:
    """Token <-> id mapping. Construct with ``Vocab.build`` or ``from_dict``."""

    def __init__(self, itos: Sequence[str]):
        if list(itos[: len(SPECIALS)]) != SPECIALS:
            raise ValueError(f"vocab must start with {SPECIALS}")
        if len(set(itos)) != len(itos):
            raise ValueError("vocab contains duplicate tokens")
        self.itos: List[str] = list(itos)
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(
        cls,
        token_lists: Iterable[Sequence[str]],
        max_size: int = MAX_VOCAB,
        min_freq: int = MIN_FREQ,
    ) -> "Vocab":
        """Count tokens and keep the most frequent ones (deterministic order)."""
        if max_size < len(SPECIALS):
            raise ValueError(f"max_size must be >= {len(SPECIALS)}")
        counts: Counter = Counter()
        for tokens in token_lists:
            counts.update(tokens)
        for special in SPECIALS:
            counts.pop(special, None)
        ranked = sorted(
            (t for t, c in counts.items() if c >= min_freq),
            key=lambda t: (-counts[t], t),
        )
        return cls(SPECIALS + ranked[: max_size - len(SPECIALS)])

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def encode(self, tokens: Sequence[str], max_len: int = MAX_LEN) -> List[int]:
        """Map tokens to ids (unknown -> ``UNK_ID``), keeping the first ``max_len``."""
        return [self.stoi.get(t, UNK_ID) for t in tokens[:max_len]]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[i] for i in ids]

    def to_dict(self) -> Dict[str, List[str]]:
        """JSON-serializable form, stored alongside the model artifact."""
        return {"itos": list(self.itos)}

    @classmethod
    def from_dict(cls, obj: Dict[str, List[str]]) -> "Vocab":
        return cls(obj["itos"])

    def sha256(self) -> str:
        """Fingerprint of the exact id mapping, recorded with each run."""
        return hashlib.sha256(json.dumps(self.itos).encode("utf-8")).hexdigest()


def encode_text(text: str, vocab: Vocab, max_len: int = MAX_LEN) -> Tuple[List[str], List[int]]:
    """Tokenize + encode one review. Returns ``(tokens_kept, ids)``.

    Raises ``ValueError`` if the text has no tokens (an empty sequence cannot
    be packed or attended over).
    """
    tokens = tokenize(text)[:max_len]
    if not tokens:
        raise ValueError("text contains no tokens")
    return tokens, vocab.encode(tokens, max_len=max_len)


def pad_batch(
    sequences: Sequence[Sequence[int]], pad_id: int = PAD_ID
) -> Tuple[np.ndarray, np.ndarray]:
    """Right-pad id lists to the longest one. Returns ``(ids [B, T], lengths [B])``."""
    if not sequences:
        raise ValueError("cannot pad an empty batch")
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    if (lengths == 0).any():
        raise ValueError("every sequence needs at least one token")
    ids = np.full((len(sequences), int(lengths.max())), pad_id, dtype=np.int64)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = seq
    return ids, lengths


def padding_mask(lengths: Sequence[int], max_len: int | None = None) -> np.ndarray:
    """Boolean ``[B, T]``: True where position ``t < length`` (a real token)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    t = int(lengths.max()) if max_len is None else max_len
    return np.arange(t)[None, :] < lengths[:, None]
