"""The two Seq-Sight sentiment classifiers, sharing one packed-LSTM encoder.

    ids [B, T], lengths [B]
      -> Embedding (pad id 0 -> zero vector) -> Dropout
      -> LSTM over a *packed* sequence (padding is never fed through the cell)
      -> outputs [B, T, H] (zeros at padded steps), last hidden state [B, H]

    baseline  "lstm":       last hidden state             -> Dropout -> Linear(H, 2)
    attention "attention":  AttentionPooling(outputs, mask) -> Dropout -> Linear(H, 2)

The only difference between the two is how the sequence is summarized: the
baseline relies on whatever survived in the final recurrent state, while the
attention model can read every timestep directly. Everything else (embedding,
LSTM, head, dropout) is identical, which keeps the comparison matched.

Both ``forward`` methods return ``(logits [B, 2], attention [B, T] or None)``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from seqsight.attention import AttentionPooling
from shared.imdb import N_CLASSES
from shared.text import PAD_ID, pad_batch

MODEL_KINDS = ("lstm", "attention")

DEFAULT_CONFIG = {
    "kind": "attention",
    "vocab_size": 20_000,
    "embed_dim": 128,
    "hidden_dim": 128,
    "attn_dim": 64,
    "bidirectional": False,
    "dropout": 0.3,
}


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Boolean ``[B, max_len]``, True on real tokens (torch twin of ``shared.text.padding_mask``)."""
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def collate(sequences: Sequence[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch of id lists into ``(ids [B, T], lengths [B])`` tensors."""
    ids, lengths = pad_batch(sequences)
    return torch.from_numpy(ids), torch.from_numpy(lengths)


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 bidirectional: bool = False, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(outputs [B, T, out_dim], last [B, out_dim])``.

        ``last`` is the hidden state after each sequence's own final *real* token
        (packing stops every sequence at its length; ``h_n`` comes back in the
        original batch order even with ``enforce_sorted=False``).
        """
        emb = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True,
                                         total_length=ids.size(1))
        last = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.bidirectional else h_n[-1]
        return outputs, last


class LSTMClassifier(nn.Module):
    """Baseline: classify from the LSTM's final hidden state."""

    kind = "lstm"

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 bidirectional: bool = False, dropout: float = 0.0, **_):
        super().__init__()
        self.encoder = LSTMEncoder(vocab_size, embed_dim, hidden_dim, bidirectional, dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.out_dim, N_CLASSES))

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, None]:
        _, last = self.encoder(ids, lengths)
        return self.head(last), None


class AttentionLSTMClassifier(nn.Module):
    """LSTM + scaled-dot-product attention pooling over every timestep."""

    kind = "attention"

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 bidirectional: bool = False, dropout: float = 0.0,
                 attn_dim: int = 64, **_):
        super().__init__()
        self.encoder = LSTMEncoder(vocab_size, embed_dim, hidden_dim, bidirectional, dropout)
        self.attention = AttentionPooling(self.encoder.out_dim, attn_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.out_dim, N_CLASSES))

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.encoder(ids, lengths)
        mask = lengths_to_mask(lengths.to(ids.device), ids.size(1))
        context, weights = self.attention(outputs, mask)
        return self.head(context), weights


def build_model(config: dict) -> nn.Module:
    """Instantiate the model a run config describes (``config["kind"]`` picks it)."""
    cfg = {**DEFAULT_CONFIG, **config}
    if cfg["kind"] not in MODEL_KINDS:
        raise ValueError(f"kind must be one of {MODEL_KINDS}")
    cls = AttentionLSTMClassifier if cfg["kind"] == "attention" else LSTMClassifier
    return cls(**{k: v for k, v in cfg.items() if k != "kind"})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_ids(model: nn.Module, ids: List[int]) -> Tuple[List[float], Optional[List[float]]]:
    """Class probabilities and (for the attention model) per-token weights for one review."""
    model.eval()
    x, lengths = collate([ids])
    logits, weights = model(x, lengths)
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    return probs, (weights[0].tolist() if weights is not None else None)


def load_exported(stem: str | Path) -> Tuple[nn.Module, dict]:
    """Load ``<stem>.pt`` (a plain state_dict) and ``<stem>.json`` (the run config)."""
    stem = Path(stem)
    config = json.loads(stem.with_suffix(".json").read_text())
    model = build_model(config)
    model.load_state_dict(torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True))
    model.eval()
    return model, config
