"""Bootstrap intervals: they bracket the point estimate, shrink with n, and pair correctly."""
from __future__ import annotations

import numpy as np

from src.metrics import evaluate, macro_f1, per_class, recall_of


def _noisy(y, err, rng):
    p = y.copy()
    flip = rng.random(len(y)) < err
    p[flip] = rng.integers(0, 8, flip.sum())
    return p


def test_interval_brackets_point_and_is_deterministic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 8, 600)
    p = _noisy(y, 0.2, rng)
    a = evaluate(y, p, 8, priority=6, n_boot=300)
    b = evaluate(y, p, 8, priority=6, n_boot=300)
    assert a == b
    for k in ("accuracy", "macro_f1", "priority_recall"):
        lo, hi = a[f"{k}_ci"]
        assert lo <= a[k] <= hi


def test_interval_narrows_with_more_data():
    rng = np.random.default_rng(1)
    widths = []
    for n in (200, 3000):
        y = rng.integers(0, 8, n)
        lo, hi = evaluate(y, _noisy(y, 0.3, rng), 8, 6, n_boot=300)["macro_f1_ci"]
        widths.append(hi - lo)
    assert widths[1] < widths[0] / 2


def test_paired_delta_detects_a_real_gap():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 8, 1000)
    good, bad = _noisy(y, 0.1, rng), _noisy(y, 0.5, rng)
    ev = evaluate(y, good, 8, 6, baseline=bad, n_boot=300)
    assert ev["delta_macro_f1"] > 0.3
    assert ev["delta_macro_f1_ci"][0] > 0          # interval excludes zero
    same = evaluate(y, good, 8, 6, baseline=good, n_boot=300)
    assert same["delta_macro_f1"] == 0 and same["delta_macro_f1_ci"] == (0.0, 0.0)


def test_helpers():
    y = np.array([0, 0, 1, 1, 6, 6])
    p = np.array([0, 1, 1, 1, 6, 0])
    assert recall_of(y, p, 6) == 0.5
    assert 0 < macro_f1(y, p, 8) < 1
    pc = per_class(y, p, [f"q{i}" for i in range(8)])
    assert pc["q6"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}


def test_fast_macro_f1_matches_sklearn():
    from src.metrics import macro_f1_sklearn
    rng = np.random.default_rng(3)
    for _ in range(20):
        y = rng.integers(0, 8, 300)
        p = _noisy(y, rng.random(), rng)
        p[p == 7] = 0                                  # a class that is never predicted
        assert abs(macro_f1(y, p, 8) - macro_f1_sklearn(y, p, 8)) < 1e-12
