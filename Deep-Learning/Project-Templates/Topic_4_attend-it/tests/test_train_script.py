"""Tests for the offline training loop in scripts/train.py (tiny data, seconds)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.train import evaluate, macro_f1, predict, train_model

SMALL = {"vocab_size": 30, "embed_dim": 8, "hidden_dim": 8, "attn_dim": 4, "dropout": 0.0}


def _toy(n: int, seed: int):
    # Label 1 iff token 3 appears; lengths vary 3..10.
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        s = rng.integers(4, 30, size=int(rng.integers(3, 11))).tolist()
        y = int(rng.integers(0, 2))
        if y:
            s[int(rng.integers(0, len(s)))] = 3
        data.append((s, y))
    return data


def test_macro_f1():
    y = np.array([0, 0, 1, 1])
    assert macro_f1(y, y) == 1.0
    assert macro_f1(y, np.array([0, 0, 0, 0])) == pytest.approx((2 / 3 + 0) / 2)


@pytest.mark.parametrize("kind", ["lstm", "attention"])
def test_train_model_learns_and_keeps_best_epoch(kind):
    train, val = _toy(200, 0), _toy(60, 1)
    model, history, best = train_model({**SMALL, "kind": kind}, train, val,
                                       epochs=8, batch_size=32, lr=0.02, log=lambda *_: None)
    assert len(history) == 8 and 1 <= best <= 8
    assert history[best - 1]["val_acc"] == max(h["val_acc"] for h in history)
    # The returned model is the best-epoch checkpoint, not the last epoch.
    assert evaluate(model, val)["acc"] == pytest.approx(history[best - 1]["val_acc"])
    assert history[best - 1]["val_acc"] > 0.8


def test_training_is_reproducible():
    train, val = _toy(64, 0), _toy(16, 1)
    runs = [train_model({**SMALL, "kind": "attention"}, train, val, epochs=2,
                        batch_size=16, lr=0.01, log=lambda *_: None) for _ in range(2)]
    for a, b in zip(runs[0][0].state_dict().values(), runs[1][0].state_dict().values()):
        assert torch.equal(a, b)


def test_predict_returns_unpadded_attention_in_order():
    torch.manual_seed(0)
    from seqsight.model import build_model

    model = build_model({**SMALL, "kind": "attention"})
    data = [([5, 6, 7], 0), ([8], 1), ([9, 10], 0)]
    probs, attn = predict(model, data, batch_size=2)
    assert probs.shape == (3, 2)
    assert [len(a) for a in attn] == [3, 1, 2]
    assert all(sum(a) == pytest.approx(1.0) for a in attn)
    assert predict(build_model({**SMALL, "kind": "lstm"}), data)[1] is None
