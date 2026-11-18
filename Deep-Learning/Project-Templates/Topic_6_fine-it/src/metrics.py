"""Headline metrics with 95% bootstrap confidence intervals.

Every interval in the repo comes from here: resample the test set with
replacement B times, recompute the metric, take the 2.5th and 97.5th
percentiles (the percentile bootstrap). The Δ against the baseline is PAIRED --
both models are scored on the same resample each time -- which is what makes a
small Δ detectable: the per-message difficulty cancels out.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

N_BOOT = 2000
BOOT_SEED = 1234


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float((y == p).mean())


def macro_f1(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1 (a class with no support and no predictions scores 0,
    matching sklearn's ``zero_division=0``). A confusion-matrix version, ~50x faster than
    sklearn, because the bootstrap calls it thousands of times."""
    cm = np.bincount(np.asarray(y) * n_classes + np.asarray(p), minlength=n_classes * n_classes)
    cm = cm.reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)                     # = 2tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def macro_f1_sklearn(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    return float(f1_score(y, p, labels=list(range(n_classes)), average="macro", zero_division=0))


def recall_of(y: np.ndarray, p: np.ndarray, cls: int) -> float:
    m = y == cls
    return float((p[m] == cls).mean()) if m.any() else float("nan")


def _ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(y: np.ndarray, p: np.ndarray, n_classes: int, priority: int,
             baseline: np.ndarray | None = None, n_boot: int = N_BOOT,
             seed: int = BOOT_SEED) -> dict:
    """Point estimates + percentile-bootstrap CIs; paired Δ macro-F1 when ``baseline`` is given."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    acc, f1, rec, delta = [], [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        acc.append(accuracy(y[i], p[i]))
        f1.append(macro_f1(y[i], p[i], n_classes))
        rec.append(recall_of(y[i], p[i], priority))
        if baseline is not None:
            delta.append(f1[-1] - macro_f1(y[i], baseline[i], n_classes))
    out = {
        "accuracy": accuracy(y, p), "accuracy_ci": _ci(np.array(acc)),
        "macro_f1": macro_f1(y, p, n_classes), "macro_f1_ci": _ci(np.array(f1)),
        "priority_recall": recall_of(y, p, priority), "priority_recall_ci": _ci(np.array(rec)),
        "n": int(n), "n_boot": n_boot, "boot_seed": seed,
    }
    if baseline is not None:
        out["delta_macro_f1"] = out["macro_f1"] - macro_f1(y, np.asarray(baseline), n_classes)
        out["delta_macro_f1_ci"] = _ci(np.array(delta))
    return out


def per_class(y: np.ndarray, p: np.ndarray, names: list[str]) -> dict:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(len(names))), zero_division=0)
    return {n: {"precision": round(float(a), 4), "recall": round(float(b), 4),
                "f1": round(float(c), 4), "support": int(d)}
            for n, a, b, c, d in zip(names, pr, rc, f1, sup)}
