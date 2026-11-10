"""A transformer encoder written out by hand, with explicit shape annotations.

Notation used in every docstring and shape comment:

* ``B`` batch size, ``T`` sequence length (``Tq`` queries, ``Tk`` keys),
* ``D`` model width (``d_model``), ``H`` number of heads,
* ``d_h = D / H`` the width of one head, ``F`` the MLP hidden width (``d_ff``).

What lives here
---------------
``scaled_dot_product_attention``
    ``softmax(Q Kᵀ / √d_k + M) V``. The softmax runs over the key axis, so every
    query's attention weights sum to 1.
``MultiHeadAttention``
    ``H`` scaled-dot-product attentions run in parallel on ``d_h``-wide slices of
    linear projections of the input; their outputs are concatenated back to
    width ``D`` and mixed by an output projection ``W_O``::

        head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)        [B, T, d_h]
        MHA(X) = Concat(head_1, ..., head_H) W_O             [B, T, D]

    The parameter layout (one fused ``in_proj_weight [3D, D]`` plus
    ``out_proj``) deliberately matches ``torch.nn.MultiheadAttention`` so weights
    can be copied across for matched-initialisation benchmarks.
``sinusoidal_positional_encoding`` / ``SinusoidalPositionalEncoding``
    ``PE[p, 2i] = sin(p / 10000^(2i/D))``, ``PE[p, 2i+1] = cos(p / 10000^(2i/D))``.
``TransformerEncoderBlock``
    Pre-norm block: ``x + MHA(LN₁(x))`` then ``x + MLP(LN₂(x))``. Parameter names
    match ``torch.nn.TransformerEncoderLayer(norm_first=True)``.
``TinyTransformerEncoder``
    Token embedding (+ optional positional encoding) followed by a stack of
    blocks. Used by the ``/encoder_attention`` endpoint and by the
    permutation-equivariance test.

Why position matters: self-attention is a function of the *set* of token
vectors — permuting the input rows permutes the output rows the same way
(permutation equivariance). Word order is only visible to the model once a
positional encoding is added to the embeddings.

Masks use one convention throughout: a boolean tensor where ``True`` means
"this key may be attended to". Every query must keep at least one key.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

__all__ = [
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "sinusoidal_positional_encoding",
    "SinusoidalPositionalEncoding",
    "TransformerEncoderBlock",
    "TinyTransformerEncoder",
    "copy_block_to_torch",
    "copy_block_from_torch",
]


# ---------------------------------------------------------------------------
# Scaled dot-product attention
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(output, weights)`` for ``softmax(Q Kᵀ / √d_k) V``.

    Args:
        q: Queries ``[..., Tq, d_k]``.
        k: Keys ``[..., Tk, d_k]``.
        v: Values ``[..., Tk, d_v]``.
        mask: Optional boolean tensor broadcastable to ``[..., Tq, Tk]``;
            ``True`` marks keys that may be attended to.
        dropout_p: Dropout applied to the weights before they multiply ``V``
            (pass 0 at inference).

    Returns:
        ``output [..., Tq, d_v]`` and ``weights [..., Tq, Tk]`` (before dropout)
        whose last axis sums to 1.

    Dividing by ``√d_k`` keeps the scores at unit variance: if the entries of a
    query and a key are independent with mean 0 and variance 1, their dot product
    has variance ``d_k``. Unscaled, large ``d_k`` saturates the softmax (one
    weight ≈ 1) and its gradient vanishes.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)          # [..., Tq, Tk]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)                     # rows sum to 1
    dropped = torch.dropout(weights, dropout_p, True) if dropout_p > 0 else weights
    return dropped @ v, weights                                 # [..., Tq, d_v]


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention, ``[B, T, D] -> [B, T, D]``.

    Args:
        d_model: Model width ``D``.
        n_heads: Number of heads ``H``; must divide ``D``.
        dropout: Dropout on the attention weights (0 disables it).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # W_Q, W_K, W_V stacked as one [3D, D] matrix (same layout as torch.nn).
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)             # W_O
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-uniform projections and zero biases, as in ``nn.MultiheadAttention``."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, D] -> [B, H, T, d_h]``."""
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attend over ``x``.

        Args:
            x: Input ``[B, T, D]``.
            mask: Optional boolean mask broadcastable to ``[B, H, T, T]``
                (``True`` = may attend).

        Returns:
            ``output [B, T, D]`` and per-head ``weights [B, H, T, T]``.
        """
        b, t, d = x.shape
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias        # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                              # 3 x [B, T, D]
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        heads, weights = scaled_dot_product_attention(
            q, k, v, mask, dropout_p=self.dropout.p if self.training else 0.0
        )                                                           # [B, H, T, d_h]
        concat = heads.transpose(1, 2).reshape(b, t, d)              # [B, T, H·d_h = D]
        return self.out_proj(concat), weights                       # [B, T, D]


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Return the fixed sinusoidal table ``PE [max_len, d_model]``.

    ``PE[p, 2i] = sin(p · ω_i)`` and ``PE[p, 2i+1] = cos(p · ω_i)`` with
    ``ω_i = 10000^(-2i / d_model)``. Each pair of columns is a rotation at its own
    frequency, so ``PE[p + k]`` is a fixed linear function of ``PE[p]`` — relative
    offsets are easy for attention to read off. ``d_model`` must be even.
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for sinusoidal positional encoding")
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)          # [L, 1]
    omega = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )                                                                       # [D/2]
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * omega)
    pe[:, 1::2] = torch.cos(pos * omega)
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """Adds the sinusoidal table to an input: ``[B, T, D] -> [B, T, D]``."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.register_buffer("pe", sinusoidal_positional_encoding(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x + PE[:T]`` for ``x [B, T, D]``."""
        t = x.size(1)
        if t > self.pe.size(0):
            raise ValueError(f"sequence length {t} exceeds max_len {self.pe.size(0)}")
        return x + self.pe[:t]


# ---------------------------------------------------------------------------
# Encoder block
# ---------------------------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block, ``[B, T, D] -> [B, T, D]``.

    ::

        h   = x + Dropout(MHA(LN₁(x)))                  [B, T, D]
        out = h + Dropout(W₂ · ReLU(W₁ · LN₂(h)))       [B, T, D]   (W₁: D→F, W₂: F→D)

    Pre-norm (normalise *before* each sub-layer) keeps an un-normalised residual
    path from input to output, which is what lets deep stacks train without
    learning-rate warm-up. Parameter names match
    ``nn.TransformerEncoderLayer(norm_first=True)``.

    Args:
        d_model: Model width ``D``.
        n_heads: Number of heads ``H``.
        d_ff: MLP hidden width ``F``.
        dropout: Dropout rate on attention weights and both residual branches.
        layer_norm_eps: Epsilon of both LayerNorms.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)       # inside the MLP
        self.dropout1 = nn.Dropout(dropout)      # attention residual branch
        self.dropout2 = nn.Dropout(dropout)      # MLP residual branch

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the block.

        Args:
            x: Input ``[B, T, D]``.
            padding_mask: Optional boolean ``[B, T]``; ``True`` marks real
                tokens, ``False`` marks padding that no query may attend to.

        Returns:
            ``output [B, T, D]`` and attention ``weights [B, H, T, T]``.
        """
        mask = None
        if padding_mask is not None:
            mask = padding_mask[:, None, None, :]                     # [B, 1, 1, T]
        attn_out, weights = self.self_attn(self.norm1(x), mask)       # [B, T, D]
        h = x + self.dropout1(attn_out)                               # residual 1
        ff = self.linear2(self.dropout(torch.relu(self.linear1(self.norm2(h)))))  # [B, T, D]
        return h + self.dropout2(ff), weights                         # residual 2


# ---------------------------------------------------------------------------
# A tiny encoder stack (embedding -> optional PE -> N blocks)
# ---------------------------------------------------------------------------
class TinyTransformerEncoder(nn.Module):
    """Token ids ``[B, T]`` -> hidden states ``[B, T, D]`` plus per-layer attention.

    Args:
        vocab_size: Number of token ids.
        d_model: Model width ``D``.
        n_heads: Heads per block ``H``.
        d_ff: MLP hidden width ``F``.
        n_layers: Number of stacked blocks.
        max_len: Longest sequence the positional table covers.
        use_positional_encoding: Add sinusoidal positions to the embeddings.
            Without them the encoder is permutation-equivariant (blind to order).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
        n_layers: int = 2,
        max_len: int = 128,
        use_positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len) if use_positional_encoding else None
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)         # final norm, standard for pre-norm stacks
        self.scale = math.sqrt(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        add_positional_encoding: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode ``tokens [B, T]``.

        Args:
            tokens: Token ids ``[B, T]``.
            padding_mask: Optional boolean ``[B, T]``, ``True`` = real token.
            add_positional_encoding: Set ``False`` to skip the positional table
                for this call (a teaching switch: the same weights become blind
                to word order and therefore permutation-equivariant).

        Returns:
            ``hidden [B, T, D]`` and a list of ``n_layers`` attention tensors,
            each ``[B, H, T, T]``.
        """
        x = self.embed(tokens) * self.scale                          # [B, T, D]
        if self.pos is not None and add_positional_encoding:
            x = self.pos(x)
        attentions: List[torch.Tensor] = []
        for block in self.blocks:
            x, w = block(x, padding_mask)
            attentions.append(w)
        return self.norm(x), attentions


# ---------------------------------------------------------------------------
# Weight transfer to/from torch.nn (matched initialisation)
# ---------------------------------------------------------------------------
_SHARED_KEYS = (
    "self_attn.in_proj_weight", "self_attn.in_proj_bias",
    "self_attn.out_proj.weight", "self_attn.out_proj.bias",
    "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
    "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
)


def copy_block_to_torch(block: TransformerEncoderBlock, layer: nn.TransformerEncoderLayer) -> None:
    """Copy ``block``'s parameters into ``layer`` (built with ``norm_first=True, batch_first=True``)."""
    src, dst = block.state_dict(), layer.state_dict()
    with torch.no_grad():
        for key in _SHARED_KEYS:
            dst[key].copy_(src[key])


def copy_block_from_torch(layer: nn.TransformerEncoderLayer, block: TransformerEncoderBlock) -> None:
    """Copy ``layer``'s parameters into ``block`` (the inverse of ``copy_block_to_torch``)."""
    src, dst = layer.state_dict(), block.state_dict()
    with torch.no_grad():
        for key in _SHARED_KEYS:
            dst[key].copy_(src[key])
