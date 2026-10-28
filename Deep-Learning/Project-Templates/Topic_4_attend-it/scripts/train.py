"""Train one Seq-Sight model offline on IMDB and export it into models/.

Both models are trained with the SAME seed, vocab, data order, epochs, batch
size, learning rate, max length and thread count, so the only difference
between the two runs is the attention block:

    python -m scripts.train --kind lstm      --threads 6
    python -m scripts.train --kind attention --threads 6

Writes ``models/<kind>.pt`` (state_dict of the best-validation epoch),
``models/<kind>.json`` (run config + metrics + per-epoch history) and
``models/vocab.json`` (shared by both runs; a second run checks it matches).
Then run ``python -m scripts.analyze`` for the comparison and Failure Gallery.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
import uuid
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from seqsight import MODELS_DIR
from seqsight.model import DEFAULT_CONFIG, build_model, collate, count_parameters
from shared.imdb import CLASS_NAMES, N_CLASSES, load_split, train_val_split
from shared.text import MAX_LEN, MAX_VOCAB, MIN_FREQ, Vocab, encode_text, tokenize

Example = Tuple[List[int], int]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in range(N_CLASSES):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def predict(model: nn.Module, data: Sequence[Example], batch_size: int = 256
            ) -> Tuple[np.ndarray, List[List[float]] | None]:
    """Return ``(probs [N, C], attention per example or None)`` in input order."""
    model.eval()
    probs, attn = [], []
    for s in range(0, len(data), batch_size):
        chunk = data[s:s + batch_size]
        ids, lengths = collate([c[0] for c in chunk])
        logits, w = model(ids, lengths)
        probs.append(torch.softmax(logits, -1).numpy())
        if w is not None:
            attn.extend(w[i, :n].tolist() for i, n in enumerate(lengths.tolist()))
    return np.concatenate(probs), (attn if attn else None)


def evaluate(model: nn.Module, data: Sequence[Example]) -> Dict[str, float]:
    probs, _ = predict(model, data)
    y = np.array([c[1] for c in data])
    pred = probs.argmax(1)
    loss = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())
    return {"loss": loss, "acc": float((pred == y).mean()), "macro_f1": macro_f1(y, pred)}


def train_model(config: dict, train: Sequence[Example], val: Sequence[Example],
                epochs: int, batch_size: int, lr: float, seed: int = 0, log=print
                ) -> Tuple[nn.Module, List[dict], int]:
    """Adam + cross-entropy, gradient clipping at 1.0, keep the best-val epoch.

    Returns ``(best_model, history, best_epoch)``; ``best_epoch`` is 1-based.
    """
    torch.manual_seed(seed)
    model = build_model(config)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history: List[dict] = []
    best_state, best_acc, best_epoch = None, -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        order = rng.permutation(len(train))
        for s in range(0, len(order), batch_size):
            chunk = [train[i] for i in order[s:s + batch_size]]
            ids, lengths = collate([c[0] for c in chunk])
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(ids, lengths)[0],
                                               torch.tensor([c[1] for c in chunk]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(chunk)
        val_metrics = evaluate(model, val)
        row = {"epoch": epoch, "train_loss": total / len(train), "val_loss": val_metrics["loss"],
               "val_acc": val_metrics["acc"], "seconds": time.time() - t0}
        history.append(row)
        log(f"[{config['kind']}] epoch {epoch}/{epochs} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f} ({row['seconds']:.0f}s)")
        if val_metrics["acc"] > best_acc:
            best_acc, best_epoch = val_metrics["acc"], epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def encode_all(texts: Sequence[str], labels: np.ndarray, idx: Sequence[int], vocab: Vocab
               ) -> List[Example]:
    return [(encode_text(texts[i], vocab)[1], int(labels[i])) for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["lstm", "attention"], required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--subset", type=int, default=0, help="use only N train/val/test reviews (dry runs)")
    ap.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    texts, labels = load_split("train")
    train_idx, val_idx = train_val_split(len(texts), args.val_fraction, seed=args.seed)
    test_idx = range(len(texts))
    if args.subset:   # seeded sample: the parquet files are sorted by label
        train_idx, val_idx = train_idx[:args.subset], val_idx[:args.subset]
        test_idx = np.random.default_rng(args.seed).permutation(len(texts))[:args.subset]
    vocab = Vocab.build(tokenize(texts[i]) for i in train_idx)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    vocab_path = out / "vocab.json"
    if vocab_path.exists():
        saved = Vocab.from_dict(json.loads(vocab_path.read_text()))
        if saved.sha256() != vocab.sha256():
            raise SystemExit(f"{vocab_path} differs from this run's vocab; runs would not be matched")
    else:   # atomic write: the other run may be starting at the same moment
        tmp = vocab_path.with_suffix(f".{args.kind}.tmp")
        tmp.write_text(json.dumps(vocab.to_dict()))
        tmp.replace(vocab_path)
    train = encode_all(texts, labels, train_idx, vocab)
    val = encode_all(texts, labels, val_idx, vocab)
    test_texts, test_labels = load_split("test")
    test = encode_all(test_texts, test_labels, test_idx, vocab)

    # Full architecture config, so load_exported rebuilds exactly this model.
    config = {**DEFAULT_CONFIG, "kind": args.kind, "vocab_size": len(vocab)}
    if args.kind == "lstm":
        config.pop("attn_dim")
    t0 = time.time()
    model, history, best_epoch = train_model(config, train, val, args.epochs,
                                             args.batch_size, args.lr, args.seed)
    train_seconds = time.time() - t0
    test_metrics = evaluate(model, test)
    print(f"[{args.kind}] best epoch {best_epoch}: test_acc={test_metrics['acc']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f}")

    torch.save(model.state_dict(), out / f"{args.kind}.pt")
    record = {
        **config,
        "run_id": uuid.uuid4().hex[:8],
        "classes": CLASS_NAMES,
        "max_len": MAX_LEN, "max_vocab": MAX_VOCAB, "min_freq": MIN_FREQ,
        "vocab_sha256": vocab.sha256(),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "optimizer": "adam", "grad_clip": 1.0, "seed": args.seed,
        "train_reviews": len(train), "val_reviews": len(val), "test_reviews": len(test),
        "params": count_parameters(model),
        "threads": torch.get_num_threads(),
        "hardware": f"{platform.processor() or platform.machine()} ({torch.get_num_threads()} threads)",
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "val_acc": history[best_epoch - 1]["val_acc"],
        "test_acc": test_metrics["acc"], "test_loss": test_metrics["loss"],
        "macro_f1": test_metrics["macro_f1"],
        "history": history,
    }
    (out / f"{args.kind}.json").write_text(json.dumps(record, indent=2))
    print(f"Wrote {out / (args.kind + '.pt')} and .json")


if __name__ == "__main__":
    main()
