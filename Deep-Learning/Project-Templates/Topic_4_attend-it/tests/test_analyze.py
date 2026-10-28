"""Tests for the comparison metrics in scripts/analyze.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.analyze import (
    attention_stats,
    expected_calibration_error,
    length_buckets,
    mcnemar_p,
    most_confident_errors,
    per_class,
    selective_accuracy,
    wilson_interval,
)


def test_length_buckets_split_by_raw_length_and_flag_truncation():
    rows = length_buckets([50, 100, 101, 300, 257], [True, False, True, True, False],
                          edges=(100, 256))
    assert [r["bucket"] for r in rows] == ["1-100", "101-256", ">256 (truncated)"]
    assert [r["n"] for r in rows] == [2, 1, 2]
    assert [r["acc"] for r in rows] == [0.5, 1.0, 0.5]


def test_length_buckets_empty_bucket_has_no_accuracy():
    assert length_buckets([5], [True], edges=(10,))[1] == {"bucket": ">10 (truncated)", "n": 0, "acc": None}


def test_ece_is_zero_when_confidence_matches_accuracy():
    conf = [0.75] * 4
    assert expected_calibration_error(conf, [True, True, True, False]) == pytest.approx(0.0)


def test_ece_measures_overconfidence():
    # 99% confident but only half right -> ECE ~0.49.
    assert expected_calibration_error([0.99] * 4, [True, False, True, False]) == pytest.approx(0.49)


def test_attention_stats_uniform_vs_one_hot():
    uniform = attention_stats([[0.25] * 4])
    assert uniform["mean_normalized_entropy"] == pytest.approx(1.0)
    assert uniform["median_peak_ratio"] == pytest.approx(1.0)
    one_hot = attention_stats([[1.0, 0.0, 0.0, 0.0]])
    assert one_hot["mean_normalized_entropy"] == pytest.approx(0.0)
    assert one_hot["median_peak_ratio"] == pytest.approx(4.0)


def test_attention_stats_skips_single_token_reviews():
    assert attention_stats([[1.0], [0.5, 0.5]])["mean_normalized_entropy"] == pytest.approx(1.0)


def test_most_confident_errors_orders_mistakes_by_confidence():
    conf = [0.99, 0.95, 0.6, 0.97, 0.8]
    correct = [True, False, False, False, True]
    assert most_confident_errors(conf, correct, k=2) == [3, 1]


def test_per_class_precision_recall():
    y = np.array([0, 0, 1, 1])
    out = per_class(y, np.array([0, 1, 1, 1]))
    assert out["negative"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3)}
    assert out["positive"]["precision"] == pytest.approx(2 / 3)
    assert not math.isnan(out["positive"]["f1"])


def test_wilson_interval_brackets_the_accuracy_and_narrows_with_n():
    lo, hi = wilson_interval(870, 1000)
    assert lo < 0.87 < hi and hi - lo == pytest.approx(0.0417, abs=1e-3)
    lo2, hi2 = wilson_interval(21_913, 25_000)
    assert hi2 - lo2 < 0.01


def test_mcnemar_uses_only_disagreements():
    assert mcnemar_p(100, 100) == pytest.approx(1.0, abs=0.06)   # balanced -> no evidence
    assert mcnemar_p(1351, 1099) < 1e-5                           # our two models
    assert mcnemar_p(0, 0) == 1.0


def test_selective_accuracy_trades_coverage_for_accuracy():
    conf = [0.55, 0.65, 0.95, 0.99]
    correct = [False, True, True, True]
    rows = selective_accuracy(conf, correct, thresholds=(0.6, 0.9))
    assert rows[0] == {"threshold": 0.6, "coverage": 0.75, "acc": 1.0, "abstained_acc": 0.0}
    assert rows[1]["coverage"] == 0.5 and rows[1]["abstained_acc"] == 0.5
