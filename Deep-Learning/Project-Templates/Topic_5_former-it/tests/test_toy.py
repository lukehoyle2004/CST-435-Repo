"""The toy palindrome task, the exported encoder, and the matched-init benchmark."""
from __future__ import annotations

import argparse
import json

import torch

from scripts import benchmark_block
from src.toy import (
    CONFIG_PATH,
    MAX_LEN,
    MIN_LEN,
    attention_summary,
    canonical_ids,
    is_palindrome,
    load_classifier,
    make_dataset,
    pad_batch,
)


def test_canonical_ids_relabel_by_first_appearance():
    assert canonical_ids("dog bites man bites dog".split()) == [1, 2, 3, 2, 1]
    assert canonical_ids(["x", "x", "y"]) == [1, 1, 2]
    assert canonical_ids([]) == []


def test_relabelling_preserves_palindromes():
    words = "never odd or even odd never".split()
    assert is_palindrome(words) == is_palindrome(canonical_ids(words))


def test_dataset_is_balanced_and_correctly_labelled():
    seqs, y = make_dataset(600, seed=3)
    assert y.float().mean().item() == 0.5
    for s, lab in zip(seqs, y.tolist()):
        assert MIN_LEN <= len(s) <= MAX_LEN
        assert is_palindrome(s) == bool(lab)
        assert s == canonical_ids(s)          # already canonical


def test_pad_batch_mask():
    tokens, mask = pad_batch([[1, 2], [1, 2, 3, 2, 1]], max_len=6)
    assert tokens.shape == mask.shape == (2, 6)
    assert mask.sum(1).tolist() == [2, 5]


def test_exported_classifier_regression():
    """The committed weights still score what their metadata says (to within sampling noise)."""
    model, meta = load_classifier()
    seqs, y = make_dataset(1000, seed=123)
    tokens, mask = pad_batch(seqs)
    with torch.no_grad():
        logits, attns = model(tokens, mask)
    acc = (logits.argmax(-1) == y).float().mean().item()
    assert acc >= meta["test"]["accuracy"] - 0.03
    assert len(attns) == meta["architecture"]["n_layers"]
    assert attns[0].shape == (1000, meta["architecture"]["n_heads"], MAX_LEN, MAX_LEN)


def test_metadata_matches_weights_on_disk():
    meta = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert meta["params"] == sum(p.numel() for p in load_classifier()[0].parameters())


def test_attention_summary_on_known_matrix():
    n = 4
    a = torch.zeros(1, n, n)
    for i in range(n):
        a[0, i, n - 1 - i] = 1.0                       # a perfect mirror head
    s = attention_summary([1, 2, 2, 1], a)[0]
    assert s["mirror"] == 1.0 and s["self"] == 0.0 and s["uniform"] == 0.25
    assert s["same_word"] == 1.0                        # partners hold the same word here


def test_matched_init_benchmark_losses_agree():
    args = argparse.Namespace(d_model=16, heads=2, d_ff=32, seq_len=6, batch=8, steps=25,
                              lr=1e-3, reps=3, threads=1, seed=0, no_plot=True)
    result = benchmark_block.run(args)
    assert result["initial_forward_max_abs_diff"] < 1e-5
    assert result["max_abs_loss_diff"] < 1e-5
    assert len(result["loss_hand_written"]) == 25
    assert result["loss_hand_written"][-1] < result["loss_hand_written"][0]
