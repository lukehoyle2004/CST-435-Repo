"""Compare the two exported models on the full IMDB test set and build the Failure Gallery.

Reads ``models/{lstm,attention}.{pt,json}`` and ``models/vocab.json`` (written by
scripts/train.py), scores all 25,000 test reviews with both models, and writes:

    models/runs.json      one summary row per run -> Supabase `runs` (Phase 4 seed)
    models/gallery.json   failures: the attention model's most confident mistakes
                          examples: short, correctly classified reviews for the UI
                          attention_wins / lstm_wins: reviews only one model got right

    python -m scripts.analyze

Everything here is IMDB's public test set; no user input is involved.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from scripts.train import macro_f1, predict
from seqsight import MODELS_DIR
from seqsight.model import load_exported
from shared.imdb import CLASS_NAMES, load_split
from shared.text import MAX_LEN, Vocab, tokenize

# Raw token counts (before truncation to MAX_LEN) for the length analysis.
LENGTH_EDGES = (100, 200, MAX_LEN)
CONFIDENT = 0.9
DISPLAY = {
    "lstm": ("LSTM (final hidden state)",
             "Embedding(128) -> LSTM(128), packed -> last hidden state -> Linear(2)"),
    "attention": ("LSTM + attention",
                  "Embedding(128) -> LSTM(128), packed -> scaled-dot-product attention "
                  "pooling (learned query, d_k=64, padding mask) -> Linear(2)"),
}


def length_buckets(raw_lengths: Sequence[int], correct: Sequence[bool],
                   edges: Sequence[int] = LENGTH_EDGES) -> List[Dict]:
    """Accuracy per raw-length bucket; the last bucket holds truncated reviews."""
    raw_lengths, correct = np.asarray(raw_lengths), np.asarray(correct, dtype=bool)
    bounds = [0, *edges, math.inf]
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = (raw_lengths > lo) & (raw_lengths <= hi)
        label = f"{lo + 1}-{hi}" if hi != math.inf else f">{lo} (truncated)"
        rows.append({"bucket": label, "n": int(sel.sum()),
                     "acc": float(correct[sel].mean()) if sel.any() else None})
    return rows


def expected_calibration_error(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """Weighted mean |accuracy - confidence| over equal-width confidence bins."""
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0.5, 1.0, bins + 1)       # binary: top-class confidence >= 0.5
    total = 0.0
    for i in range(bins):
        sel = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= 1.0))
        if sel.any():
            total += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(total)


def attention_stats(weights: Sequence[Sequence[float]]) -> Dict[str, float]:
    """How peaked is attention? Normalized entropy (1 = uniform, 0 = one-hot) and
    peak ratio (top weight x length; 1 = uniform)."""
    ent, peak = [], []
    for w in weights:
        w = np.asarray(w)
        if len(w) < 2:
            continue
        nz = w[w > 0]
        ent.append(float(-(nz * np.log(nz)).sum() / math.log(len(w))))
        peak.append(float(w.max() * len(w)))
    return {"mean_normalized_entropy": float(np.mean(ent)),
            "median_peak_ratio": float(np.median(peak))}


def most_confident_errors(conf: Sequence[float], correct: Sequence[bool], k: int) -> List[int]:
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=bool)
    wrong = np.flatnonzero(~correct)
    return wrong[np.argsort(-conf[wrong], kind="stable")][:k].tolist()


def per_class(y: np.ndarray, pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = int(((pred == c) & (y == c)).sum())
        p = tp / max(int((pred == c).sum()), 1)
        r = tp / max(int((y == c).sum()), 1)
        out[name] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--limit", type=int, default=0, help="score a seeded sample of N reviews (dry runs)")
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    models_dir = args.models_dir

    vocab = Vocab.from_dict(json.loads((models_dir / "vocab.json").read_text()))
    texts, y = load_split("test")
    if args.limit:
        keep = np.random.default_rng(0).permutation(len(texts))[:args.limit]
        texts, y = [texts[i] for i in keep], y[keep]
    all_tokens = [tokenize(t) for t in texts]
    raw_lengths = np.array([len(t) for t in all_tokens])
    kept = [t[:MAX_LEN] for t in all_tokens]
    data = [(vocab.encode(t), int(lab)) for t, lab in zip(kept, y)]

    results, runs = {}, []
    for kind in ("lstm", "attention"):
        model, cfg = load_exported(models_dir / kind)
        if cfg["vocab_sha256"] != vocab.sha256():
            raise SystemExit(f"{kind} was trained with a different vocab")
        probs, attn = predict(model, data)
        pred = probs.argmax(1)
        conf = probs.max(1)
        correct = pred == y
        results[kind] = {"probs": probs, "pred": pred, "conf": conf, "correct": correct, "attn": attn}
        row = {
            "run_id": cfg["run_id"], "name": kind,
            "display_name": DISPLAY[kind][0], "architecture": DISPLAY[kind][1],
            "dataset": "IMDB (Maas et al., 2011)",
            **{k: cfg[k] for k in ("epochs", "batch_size", "lr", "seed", "params", "max_len",
                                   "train_reviews", "val_reviews", "test_reviews", "hardware",
                                   "train_seconds", "best_epoch", "val_acc", "vocab_sha256")},
            "test_acc": float(correct.mean()),
            "macro_f1": macro_f1(y, pred),
            "per_class": per_class(y, pred),
            "ece": expected_calibration_error(conf, correct),
            "confident_errors": int(((conf >= CONFIDENT) & ~correct).sum()),
            "confident_error_rate": float(((conf >= CONFIDENT) & ~correct).sum() / max((~correct).sum(), 1)),
            "length_buckets": length_buckets(raw_lengths, correct),
            "history": cfg["history"],
        }
        if attn is not None:
            row["attention_stats"] = attention_stats(attn)
        runs.append(row)

    a, b = results["lstm"], results["attention"]
    agreement = {
        "both_right": int((a["correct"] & b["correct"]).sum()),
        "only_attention_right": int((~a["correct"] & b["correct"]).sum()),
        "only_lstm_right": int((a["correct"] & ~b["correct"]).sum()),
        "both_wrong": int((~a["correct"] & ~b["correct"]).sum()),
    }
    for r in runs:
        r["agreement"] = agreement

    def card(i: int) -> Dict:
        return {
            "test_index": int(i),
            "tokens": kept[i],
            "attention": [round(w, 5) for w in b["attn"][i]],
            "raw_length": int(raw_lengths[i]),
            "true": CLASS_NAMES[y[i]],
            "attention_pred": CLASS_NAMES[b["pred"][i]],
            "attention_confidence": float(b["conf"][i]),
            "lstm_pred": CLASS_NAMES[a["pred"][i]],
            "lstm_confidence": float(a["conf"][i]),
        }

    short = raw_lengths <= 60
    examples = []
    for c in range(len(CLASS_NAMES)):
        cand = np.flatnonzero(short & b["correct"] & (y == c) & (b["conf"] >= CONFIDENT))
        examples += cand[:3].tolist()
    only_attn = np.flatnonzero(~a["correct"] & b["correct"])
    only_lstm = np.flatnonzero(a["correct"] & ~b["correct"])
    gallery = {
        "failures": [card(i) for i in most_confident_errors(b["conf"], b["correct"], 12)],
        "examples": [card(i) for i in examples],
        "attention_wins": [card(i) for i in only_attn[np.argsort(raw_lengths[only_attn])][:4]],
        "lstm_wins": [card(i) for i in only_lstm[np.argsort(raw_lengths[only_lstm])][:4]],
    }

    (models_dir / "runs.json").write_text(json.dumps(runs, indent=2))
    (models_dir / "gallery.json").write_text(json.dumps(gallery, indent=1))

    for r in runs:
        print(f"\n{r['display_name']}: test_acc={r['test_acc']:.4f} macro_f1={r['macro_f1']:.4f} "
              f"ece={r['ece']:.4f} confident_errors={r['confident_errors']} best_epoch={r['best_epoch']}")
        for bkt in r["length_buckets"]:
            print(f"   {bkt['bucket']:>18}: n={bkt['n']:>5} acc={bkt['acc']:.4f}")
        if "attention_stats" in r:
            print(f"   attention: {r['attention_stats']}")
    print(f"\nagreement: {agreement}")
    print("\nFailure Gallery (attention model's most confident mistakes):")
    for f in gallery["failures"][:6]:
        top = np.argsort(f["attention"])[::-1][:5]
        print(f"   true={f['true']} pred={f['attention_pred']} conf={f['attention_confidence']:.3f} "
              f"lstm={f['lstm_pred']} top={[f['tokens'][j] for j in top]}")


if __name__ == "__main__":
    main()
