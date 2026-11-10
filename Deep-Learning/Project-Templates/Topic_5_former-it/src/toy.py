"""The toy task that gives the hand-written encoder block something to show.

``POST /encoder_attention`` runs ``TinyTransformerEncoder`` (built only from the
hand-written pieces in ``src.transformer``) on a short input and returns its
attention heatmaps. Random weights would draw noise, so the encoder is trained
offline (``scripts/train_toy_encoder.py``) on a task whose solution *is* an
attention pattern: **is this sequence of words a palindrome?** ("dog bites man
bites dog" → yes). To answer, position ``i`` has to look at position ``n-1-i``,
so a trained head lights up the anti-diagonal.

Words are mapped to ids by order of first appearance ("dog bites man bites dog"
→ ``1 2 3 2 1``). Relabelling symbols never changes whether a sequence is a
palindrome, so the model needs no vocabulary of real words, and the raw words
never leave the request.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from src.transformer import TinyTransformerEncoder

PAD = 0
MAX_LEN = 12          # longest toy input (words)
MIN_LEN = 2
VOCAB_SIZE = MAX_LEN + 1   # ids 1..MAX_LEN after canonical relabelling, 0 = padding
CLASS_NAMES = ["not a palindrome", "palindrome"]

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
WEIGHTS_PATH = MODELS_DIR / "encoder_block.pt"
CONFIG_PATH = MODELS_DIR / "encoder_block.json"


def canonical_ids(symbols: Sequence) -> List[int]:
    """Relabel symbols by first appearance: ``["b","a","b"] -> [1, 2, 1]``."""
    seen: Dict = {}
    return [seen.setdefault(s, len(seen) + 1) for s in symbols]


def is_palindrome(ids: Sequence[int]) -> bool:
    return list(ids) == list(ids)[::-1]


def pad_batch(seqs: Sequence[Sequence[int]], max_len: int = MAX_LEN) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to ``max_len``. Returns ``tokens [B, max_len]`` and ``mask [B, max_len]`` (True = real)."""
    tokens = torch.full((len(seqs), max_len), PAD, dtype=torch.long)
    for row, s in enumerate(seqs):
        tokens[row, : len(s)] = torch.tensor(s, dtype=torch.long)
    return tokens, tokens != PAD


class PalindromeClassifier(nn.Module):
    """``TinyTransformerEncoder`` + masked mean-pool + linear head (2 classes)."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, d_ff: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.config = dict(d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers)
        self.encoder = TinyTransformerEncoder(
            VOCAB_SIZE, d_model, n_heads, d_ff, n_layers, max_len=MAX_LEN, use_positional_encoding=True
        )
        self.head = nn.Linear(d_model, len(CLASS_NAMES))

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor, add_positional_encoding: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """``tokens, mask [B, T]`` -> ``logits [B, 2]`` and per-layer attention ``[B, H, T, T]``.

        With ``add_positional_encoding=False`` the classifier sees a bag of words:
        it gives the same answer for any reordering of the ids, so it cannot tell
        "a b a" (ids 1 2 1, a palindrome) from "a a b" (ids 1 1 2, not one) --
        the point the interface's toggle demonstrates.
        """
        hidden, attns = self.encoder(tokens, mask, add_positional_encoding)  # [B, T, D]
        m = mask.unsqueeze(-1).float()
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)           # [B, D]
        return self.head(pooled), attns


def sample_sequence(rng: random.Random, palindrome: bool) -> List[int]:
    """One canonical sequence; negatives are often a palindrome with one word changed."""
    n = rng.randint(MIN_LEN, MAX_LEN)
    k = rng.randint(2, n)                                 # alphabet size for this sample
    if palindrome:
        half = [rng.randint(1, k) for _ in range(n // 2)]
        mid = [rng.randint(1, k)] if n % 2 else []
        seq = half + mid + half[::-1]
    else:
        while True:
            if rng.random() < 0.6:                        # hard negative: near-miss
                half = [rng.randint(1, k) for _ in range(n // 2)]
                mid = [rng.randint(1, k)] if n % 2 else []
                seq = half + mid + half[::-1]
                i = rng.randrange(n)
                seq[i] = rng.choice([s for s in range(1, k + 2) if s != seq[i]])
            else:
                seq = [rng.randint(1, k) for _ in range(n)]
            if not is_palindrome(seq):
                break
    return canonical_ids(seq)


def make_dataset(n: int, seed: int) -> Tuple[List[List[int]], torch.Tensor]:
    rng = random.Random(seed)
    seqs, labels = [], []
    for i in range(n):
        lab = i % 2
        seqs.append(sample_sequence(rng, bool(lab)))
        labels.append(lab)
    return seqs, torch.tensor(labels)


def attention_summary(ids: Sequence[int], attention: torch.Tensor) -> List[dict]:
    """Per-head statistics for one sequence's attention ``[H, n, n]`` (real tokens only).

    For each head, the average weight a query puts on:

    * ``mirror``    its palindrome partner, position ``n-1-i`` (the anti-diagonal),
    * ``self``      itself (the diagonal),
    * ``same_word`` *other* positions holding the same word,

    next to ``uniform = 1/n``, the weight a head that ignored its input would give.
    """
    n = len(ids)
    out = []
    for h in range(attention.shape[0]):
        a = attention[h, :n, :n]
        mirror = sum(a[i, n - 1 - i].item() for i in range(n)) / n
        diag = sum(a[i, i].item() for i in range(n)) / n
        same = sum(a[i, j].item() for i in range(n) for j in range(n) if j != i and ids[j] == ids[i]) / n
        out.append({"head": h, "mirror": round(mirror, 4), "self": round(diag, 4),
                    "same_word": round(same, 4), "uniform": round(1 / n, 4)})
    return out


def load_classifier() -> Tuple[PalindromeClassifier, dict]:
    """Load the exported toy classifier and its metadata (accuracy, run config)."""
    meta = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = PalindromeClassifier(**meta["architecture"])
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model, meta
