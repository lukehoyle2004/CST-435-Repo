"""Unit tests for the hand-written transformer in ``src/transformer.py``.

The four checks the assignment requires are marked [REQUIRED a-d]; the rest pin
the implementation to closed-form values and to ``torch.nn`` references.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.transformer import (
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TinyTransformerEncoder,
    TransformerEncoderBlock,
    copy_block_from_torch,
    copy_block_to_torch,
    scaled_dot_product_attention,
    sinusoidal_positional_encoding,
)

D, H, FF = 16, 4, 32


@pytest.fixture
def block() -> TransformerEncoderBlock:
    torch.manual_seed(0)
    return TransformerEncoderBlock(D, H, FF).eval()


# --------------------------------------------------------------------- (a) shape
@pytest.mark.parametrize("b,t", [(1, 1), (2, 5), (3, 12)])
def test_block_output_shape_matches_input_shape(block, b, t):  # [REQUIRED a]
    x = torch.randn(b, t, D)
    out, w = block(x)
    assert out.shape == x.shape
    assert w.shape == (b, H, t, t)


def test_encoder_stack_shapes():
    enc = TinyTransformerEncoder(vocab_size=20, d_model=D, n_heads=H, d_ff=FF, n_layers=3)
    tokens = torch.randint(0, 20, (2, 7))
    hidden, attns = enc(tokens)
    assert hidden.shape == (2, 7, D)
    assert len(attns) == 3 and all(a.shape == (2, H, 7, 7) for a in attns)


# ------------------------------------------------------- (b) attention sums to 1
def test_attention_weights_sum_to_one_over_keys(block):  # [REQUIRED b]
    _, w = block(torch.randn(2, 6, D))
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, H, 6), atol=1e-6)
    assert (w >= 0).all()


def test_padding_gets_zero_weight_and_rows_still_sum_to_one(block):
    pad = torch.tensor([[True] * 6, [True, True, True, False, False, False]])
    _, w = block(torch.randn(2, 6, D), pad)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, H, 6), atol=1e-6)
    assert torch.all(w[1, :, :, 3:] == 0)


# ------------------------------------------------------------ (c) gradient flow
def test_gradient_flows_to_every_parameter_and_the_input():  # [REQUIRED c]
    torch.manual_seed(1)
    enc = TinyTransformerEncoder(vocab_size=10, d_model=D, n_heads=H, d_ff=FF, n_layers=2)
    hidden, _ = enc(torch.randint(0, 10, (4, 8)))
    hidden.pow(2).mean().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"zero gradient for {name}"


def test_gradient_reaches_block_input(block):
    x = torch.randn(2, 5, D, requires_grad=True)
    out, _ = block(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


# ------------------------------------------------- (d) permutation equivariance
def test_block_is_permutation_equivariant_without_positional_encoding(block):  # [REQUIRED d]
    x = torch.randn(2, 7, D)
    perm = torch.randperm(7, generator=torch.Generator().manual_seed(3))
    out, _ = block(x)
    out_perm, _ = block(x[:, perm])
    assert torch.allclose(out_perm, out[:, perm], atol=1e-5)


def test_encoder_without_pe_is_equivariant_and_with_pe_is_not():  # [REQUIRED d]
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    perm = torch.tensor([5, 3, 1, 0, 2, 4])

    torch.manual_seed(0)
    no_pe = TinyTransformerEncoder(10, D, H, FF, use_positional_encoding=False).eval()
    a, _ = no_pe(tokens)
    b, _ = no_pe(tokens[:, perm])
    assert torch.allclose(b, a[:, perm], atol=1e-5)

    torch.manual_seed(0)
    with_pe = TinyTransformerEncoder(10, D, H, FF, use_positional_encoding=True).eval()
    a, _ = with_pe(tokens)
    b, _ = with_pe(tokens[:, perm])
    assert not torch.allclose(b, a[:, perm], atol=1e-3)


# ------------------------------------------------------------ closed-form checks
def test_sdpa_matches_hand_computed_example():
    q = torch.tensor([[[1.0, 1.0]]])
    k = torch.tensor([[[1.0, 1.0], [1.0, -1.0], [-1.0, 0.0]]])
    v = torch.tensor([[[10.0], [20.0], [30.0]]])
    out, w = scaled_dot_product_attention(q, k, v)
    scores = np.array([2.0, 0.0, -1.0]) / math.sqrt(2)
    expected = np.exp(scores) / np.exp(scores).sum()
    assert np.allclose(w[0, 0].numpy(), expected, atol=1e-6)
    assert np.isclose(out.item(), expected @ np.array([10.0, 20.0, 30.0]), atol=1e-4)


def test_sdpa_matches_torch_with_mask():
    torch.manual_seed(2)
    q, k, v = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 5)
    mask = torch.arange(6)[None, None, None, :] < torch.tensor([6, 2])[:, None, None, None]
    out, _ = scaled_dot_product_attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(out, ref, atol=1e-6)


def test_positional_encoding_values():
    pe = sinusoidal_positional_encoding(50, 8)
    assert pe.shape == (50, 8)
    assert torch.allclose(pe[0], torch.tensor([0.0, 1.0] * 4))
    assert pe.abs().max() <= 1.0
    p, i = 7, 1
    omega = 10000 ** (-2 * i / 8)
    assert math.isclose(pe[p, 2 * i].item(), math.sin(p * omega), abs_tol=1e-6)
    assert math.isclose(pe[p, 2 * i + 1].item(), math.cos(p * omega), abs_tol=1e-6)


def test_positional_encoding_rejects_odd_width_and_long_input():
    with pytest.raises(ValueError):
        sinusoidal_positional_encoding(10, 7)
    with pytest.raises(ValueError):
        SinusoidalPositionalEncoding(8, max_len=4)(torch.zeros(1, 5, 8))


def test_mha_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(10, 3)


# ------------------------------------------------------ torch.nn reference match
def test_mha_matches_torch_multihead_attention():
    torch.manual_seed(4)
    mine = MultiHeadAttention(D, H).eval()
    ref = nn.MultiheadAttention(D, H, batch_first=True).eval()
    ref.load_state_dict(mine.state_dict())
    x = torch.randn(2, 5, D)
    out, w = mine(x)
    ref_out, ref_w = ref(x, x, x, average_attn_weights=False)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(w, ref_w, atol=1e-6)


def test_block_matches_torch_encoder_layer_both_directions(block):
    ref = nn.TransformerEncoderLayer(
        D, H, FF, dropout=0.0, batch_first=True, norm_first=True
    ).eval()
    copy_block_to_torch(block, ref)
    x = torch.randn(3, 9, D)
    with torch.no_grad():
        assert torch.allclose(block(x)[0], ref(x), atol=1e-5)

    torch.manual_seed(9)
    ref2 = nn.TransformerEncoderLayer(D, H, FF, dropout=0.0, batch_first=True, norm_first=True).eval()
    copy_block_from_torch(ref2, block)
    with torch.no_grad():
        assert torch.allclose(block(x)[0], ref2(x), atol=1e-5)


def test_padding_mask_matches_torch_key_padding_mask(block):
    ref = nn.TransformerEncoderLayer(D, H, FF, dropout=0.0, batch_first=True, norm_first=True).eval()
    copy_block_to_torch(block, ref)
    x = torch.randn(2, 6, D)
    keep = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    with torch.no_grad():
        mine, _ = block(x, keep)
        theirs = ref(x, src_key_padding_mask=~keep)
    # torch's fast path may zero padded *query* rows; compare real positions only.
    assert torch.allclose(mine[0], theirs[0], atol=1e-5)
    assert torch.allclose(mine[1, :4], theirs[1, :4], atol=1e-5)


def test_dropout_is_active_in_train_mode_only():
    torch.manual_seed(5)
    blk = TransformerEncoderBlock(D, H, FF, dropout=0.5)
    x = torch.randn(2, 6, D)
    blk.train()
    a, _ = blk(x)
    b, _ = blk(x)
    assert not torch.allclose(a, b)
    blk.eval()
    c, _ = blk(x)
    d, _ = blk(x)
    assert torch.allclose(c, d)


def test_positional_encoding_can_be_skipped_per_call():
    torch.manual_seed(0)
    enc = TinyTransformerEncoder(10, D, H, FF, use_positional_encoding=True).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    perm = torch.tensor([3, 1, 0, 2])
    a, _ = enc(tokens, add_positional_encoding=False)
    b, _ = enc(tokens[:, perm], add_positional_encoding=False)
    assert torch.allclose(b, a[:, perm], atol=1e-5)          # same weights, now order-blind
