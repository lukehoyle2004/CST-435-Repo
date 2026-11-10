"""Train the toy word-palindrome classifier served by ``POST /encoder_attention``.

    python -m scripts.train_toy_encoder

Writes ``models/encoder_block.pt`` (weights, ~40 KB) and
``models/encoder_block.json`` (architecture, seed, held-out accuracy by length).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import torch
from torch import nn

from src.toy import (
    CONFIG_PATH,
    MAX_LEN,
    MIN_LEN,
    MODELS_DIR,
    WEIGHTS_PATH,
    PalindromeClassifier,
    make_dataset,
    pad_batch,
)


@torch.no_grad()
def evaluate(model: PalindromeClassifier, seqs, labels) -> dict:
    model.eval()
    tokens, mask = pad_batch(seqs)
    pred = model(tokens, mask)[0].argmax(-1)
    correct = pred == labels
    by_len = {}
    for n in range(MIN_LEN, MAX_LEN + 1):
        idx = [i for i, s in enumerate(seqs) if len(s) == n]
        if idx:
            by_len[n] = round(correct[idx].float().mean().item(), 4)
    return {"accuracy": round(correct.float().mean().item(), 4), "accuracy_by_length": by_len}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--train-size", type=int, default=40000)
    p.add_argument("--test-size", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    train_seqs, train_y = make_dataset(args.train_size, args.seed)
    test_seqs, test_y = make_dataset(args.test_size, args.seed + 1)
    train_tok, train_mask = pad_batch(train_seqs)

    model = PalindromeClassifier()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(args.seed)
    history = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, len(train_seqs), (args.batch,), generator=g)
        logits, _ = model(train_tok[idx], train_mask[idx])
        loss = loss_fn(logits, train_y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps:
            ev = evaluate(model, test_seqs, test_y)
            history.append({"step": step, "loss": round(loss.item(), 5), "test_acc": ev["accuracy"]})
            print(f"step {step:5d}  loss {loss.item():.4f}  test acc {ev['accuracy']:.4f}")

    final = evaluate(model, test_seqs, test_y)
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    meta = {
        "task": "word-level palindrome detection (canonical first-appearance ids)",
        "architecture": model.config,
        "params": sum(p.numel() for p in model.parameters()),
        "max_len": MAX_LEN,
        "train": {**vars(args), "seconds": round(time.perf_counter() - t0, 1)},
        "test": final,
        "history": history,
        "weights_sha256": hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest(),
    }
    CONFIG_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
