"""Scaled-dot-product attention, written out by hand (no ``F.scaled_dot_product_attention``).

    Attention(Q, K, V) = softmax( Q Kᵀ / √d_k  +  M ) V

* ``Q [B, Tq, d_k]`` queries, ``K [B, Tk, d_k]`` keys, ``V [B, Tk, d_v]`` values.
* ``Q Kᵀ`` scores every query against every key with a dot product.
* Dividing by ``√d_k`` keeps the scores at unit variance: if the entries of q
  and k are independent with mean 0 and variance 1, then ``q·k`` is a sum of
  d_k such products and has variance d_k. Without the divisor, large d_k pushes
  the softmax into saturation (one weight ≈ 1, the rest ≈ 0) where its
  gradient vanishes.
* ``M`` is the padding mask: 0 on real keys, -inf on padded keys, so
  ``exp(-inf) = 0`` and padding receives exactly zero weight.
* The softmax runs over the key axis, so each query's weights sum to 1.

``AttentionPooling`` uses this with a single *learned* query to collapse the
LSTM's hidden states ``[B, T, H]`` into one context vector ``[B, H]``. That
gives one weight per token, which is the attention bar shown in the UI.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(output [B, Tq, d_v], weights [B, Tq, Tk])``.

    ``mask`` is boolean and broadcastable to ``[B, Tq, Tk]``; True means the key
    may be attended to. Every query must have at least one True key.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)          # [B, Tq, Tk]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)                     # rows sum to 1
    return weights @ v, weights


class AttentionPooling(nn.Module):
    """One learned query attends over a sequence: ``[B, T, H] -> [B, H]``.

    Keys are a linear projection of the hidden states (``K = H W_k``); values are
    the hidden states themselves (``V = H``), so the context vector lives in the
    same space as an LSTM hidden state and the classifier head is the same size
    as the baseline's.
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.key = nn.Linear(hidden_dim, attn_dim)
        self.query = nn.Parameter(torch.randn(attn_dim) / math.sqrt(attn_dim))

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``hidden [B, T, H]``, ``mask [B, T]`` -> ``(context [B, H], weights [B, T])``."""
        q = self.query.expand(hidden.size(0), 1, -1)            # [B, 1, d_k]
        context, weights = scaled_dot_product_attention(
            q, self.key(hidden), hidden, mask.unsqueeze(1)
        )
        return context.squeeze(1), weights.squeeze(1)
