"""Unit tests for the hand-written scaled-dot-product attention block."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from seqsight.attention import AttentionPooling, scaled_dot_product_attention


def test_matches_hand_computed_example():
    # d_k = 2, one query, three keys. Scores = q·k / √2 = [2, 0, -1] / √2.
    q = torch.tensor([[[1.0, 1.0]]])
    k = torch.tensor([[[1.0, 1.0], [1.0, -1.0], [-1.0, 0.0]]])
    v = torch.tensor([[[10.0], [20.0], [30.0]]])
    out, w = scaled_dot_product_attention(q, k, v)

    scores = np.array([2.0, 0.0, -1.0]) / math.sqrt(2)
    expected_w = np.exp(scores) / np.exp(scores).sum()
    assert np.allclose(w[0, 0].numpy(), expected_w, atol=1e-6)
    assert np.isclose(out.item(), expected_w @ np.array([10.0, 20.0, 30.0]), atol=1e-5)


def test_matches_torch_reference_with_mask():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 4, 8), torch.randn(3, 6, 8), torch.randn(3, 6, 5)
    mask = torch.arange(6)[None, None, :] < torch.tensor([6, 3, 1])[:, None, None]
    out, _ = scaled_dot_product_attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(out, ref, atol=1e-6)


def test_masked_keys_get_exactly_zero_weight_and_rows_sum_to_one():
    torch.manual_seed(1)
    q, k, v = torch.randn(2, 1, 4), torch.randn(2, 5, 4), torch.randn(2, 5, 4)
    mask = torch.tensor([[True, True, True, False, False],
                         [True, False, False, False, False]]).unsqueeze(1)
    _, w = scaled_dot_product_attention(q, k, v, mask)
    assert (w[~mask.expand_as(w)] == 0).all()
    assert torch.allclose(w.sum(-1), torch.ones(2, 1))
    assert w[1, 0, 0] == 1.0     # a single real key takes all the weight


def test_sqrt_dk_scaling_keeps_scores_unit_variance():
    # q·k of unit-variance vectors has variance d_k; dividing by √d_k restores ~1.
    g = torch.Generator().manual_seed(0)
    d_k = 256
    q, k = torch.randn(20_000, d_k, generator=g), torch.randn(20_000, d_k, generator=g)
    raw = (q * k).sum(-1)
    assert raw.var().item() == pytest.approx(d_k, rel=0.05)
    assert (raw / math.sqrt(d_k)).var().item() == pytest.approx(1.0, rel=0.05)


def test_unscaled_scores_saturate_the_softmax():
    # With d_k = 256 the unscaled softmax over 10 keys is nearly one-hot; the
    # scaled one is not. This is the saturation the √d_k divisor prevents.
    g = torch.Generator().manual_seed(0)
    d_k = 256
    q, k = torch.randn(500, 1, d_k, generator=g), torch.randn(500, 10, d_k, generator=g)
    raw = q @ k.transpose(-2, -1)
    unscaled_max = torch.softmax(raw, -1).amax(-1).mean().item()
    scaled_max = torch.softmax(raw / math.sqrt(d_k), -1).amax(-1).mean().item()
    assert unscaled_max > 0.9
    assert scaled_max < 0.6


def test_attention_pooling_shapes_and_mask():
    torch.manual_seed(0)
    pool = AttentionPooling(hidden_dim=6, attn_dim=4)
    hidden = torch.randn(2, 5, 6)
    mask = torch.tensor([[True] * 5, [True, True, False, False, False]])
    context, w = pool(hidden, mask)
    assert context.shape == (2, 6) and w.shape == (2, 5)
    assert (w[~mask] == 0).all() and torch.allclose(w.sum(-1), torch.ones(2))
    # V = H, so the context is the attention-weighted average of hidden states.
    assert torch.allclose(context, (w.unsqueeze(-1) * hidden).sum(1), atol=1e-6)
