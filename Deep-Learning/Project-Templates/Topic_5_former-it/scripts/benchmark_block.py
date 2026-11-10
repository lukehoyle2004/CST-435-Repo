"""Benchmark the hand-written encoder block against ``torch.nn.TransformerEncoderLayer``.

Matched initialisation: the hand-written block is built first, then its exact
parameters are copied into a ``nn.TransformerEncoderLayer(norm_first=True,
batch_first=True)``. Both are trained on the same synthetic batches, in the
same order, with identically seeded Adam optimisers and dropout off, so any
difference in their loss curves comes from the arithmetic, not the setup.

Synthetic task ("sequence mean"): for a random input ``x [B, T, D]`` the target
at every position is ``tanh`` of the mean over the sequence — a set-level summary
that self-attention can express by spreading its weights uniformly.

Outputs (under ``docs/benchmarks/``):
    block_vs_torch.json   loss curves, max |Δloss|, single-step timings
    block_vs_torch.png    the two loss curves (overlaid) and the timing bars

Run from the project root:  python -m scripts.benchmark_block
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch
from torch import nn

from src.transformer import TransformerEncoderBlock, copy_block_to_torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "benchmarks"


def make_batches(n_steps: int, batch: int, seq_len: int, d_model: int, seed: int) -> List[tuple]:
    """Pre-generate ``n_steps`` ``(x, y)`` pairs so both models see identical data."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_steps):
        x = torch.randn(batch, seq_len, d_model, generator=g)
        y = torch.tanh(x.mean(dim=1, keepdim=True)).expand_as(x).contiguous()
        batches.append((x, y))
    return batches


def train_curve(forward: Callable[[torch.Tensor], torch.Tensor], params, batches, lr: float) -> List[float]:
    """Train with Adam + MSE over ``batches``; return the per-step loss."""
    opt = torch.optim.Adam(params, lr=lr)
    losses = []
    for x, y in batches:
        opt.zero_grad()
        loss = nn.functional.mse_loss(forward(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def time_step(forward: Callable[[torch.Tensor], torch.Tensor], params, x, y, reps: int) -> Dict[str, float]:
    """Median wall time (ms) of one forward pass and of one full training step."""
    opt = torch.optim.SGD(params, lr=0.0)          # lr 0: time the step, keep weights fixed
    for _ in range(5):                             # warm-up
        forward(x)
    fwd, step = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.no_grad():
            forward(x)
        fwd.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        opt.zero_grad()
        nn.functional.mse_loss(forward(x), y).backward()
        opt.step()
        step.append((time.perf_counter() - t0) * 1000)
    return {"forward_ms": statistics.median(fwd), "train_step_ms": statistics.median(step)}


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    mine = TransformerEncoderBlock(args.d_model, args.heads, args.d_ff, dropout=0.0)
    ref = nn.TransformerEncoderLayer(
        args.d_model, args.heads, args.d_ff, dropout=0.0, batch_first=True, norm_first=True
    )
    copy_block_to_torch(mine, ref)

    def fwd_mine(x):
        return mine(x)[0]

    def fwd_ref(x):
        return ref(x)

    batches = make_batches(args.steps, args.batch, args.seq_len, args.d_model, args.seed)
    with torch.no_grad():
        x0 = batches[0][0]
        init_diff = (fwd_mine(x0) - fwd_ref(x0)).abs().max().item()

    loss_mine = train_curve(fwd_mine, mine.parameters(), batches, args.lr)
    loss_ref = train_curve(fwd_ref, ref.parameters(), batches, args.lr)
    diffs = [abs(a - b) for a, b in zip(loss_mine, loss_ref)]

    x, y = batches[0]
    timing = {
        "hand_written": time_step(fwd_mine, mine.parameters(), x, y, args.reps),
        "torch_nn": time_step(fwd_ref, ref.parameters(), x, y, args.reps),
    }
    return {
        "config": {k: v for k, v in vars(args).items() if k != "no_plot"},
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "processor": platform.processor() or platform.machine(),
        },
        "initial_forward_max_abs_diff": init_diff,
        "loss_hand_written": loss_mine,
        "loss_torch_nn": loss_ref,
        "final_loss": {"hand_written": loss_mine[-1], "torch_nn": loss_ref[-1]},
        "max_abs_loss_diff": max(diffs),
        "mean_abs_loss_diff": sum(diffs) / len(diffs),
        "timing": timing,
    }


def plot(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    steps = range(1, len(result["loss_hand_written"]) + 1)
    ax1.plot(steps, result["loss_torch_nn"], label="torch.nn.TransformerEncoderLayer", lw=3, alpha=0.45)
    ax1.plot(steps, result["loss_hand_written"], label="hand-written block", lw=1.2, ls="--")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("MSE loss")
    ax1.set_yscale("log")
    ax1.set_title(f"Matched init: max |Δloss| = {result['max_abs_loss_diff']:.2e}")
    ax1.legend()

    names = ["hand-written", "torch.nn"]
    t = result["timing"]
    fwd = [t["hand_written"]["forward_ms"], t["torch_nn"]["forward_ms"]]
    stp = [t["hand_written"]["train_step_ms"], t["torch_nn"]["train_step_ms"]]
    xs = range(len(names))
    ax2.bar([i - 0.2 for i in xs], fwd, width=0.4, label="forward")
    ax2.bar([i + 0.2 for i in xs], stp, width=0.4, label="train step")
    ax2.set_xticks(list(xs), names)
    ax2.set_ylabel("median ms (CPU)")
    ax2.set_title("Single-step timing")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    result = run(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "block_vs_torch.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.no_plot:
        plot(result, OUT_DIR / "block_vs_torch.png")
    t = result["timing"]
    print(f"initial forward max |diff|: {result['initial_forward_max_abs_diff']:.2e}")
    print(f"final loss  hand-written {result['final_loss']['hand_written']:.6f}  "
          f"torch.nn {result['final_loss']['torch_nn']:.6f}  max |Δ| {result['max_abs_loss_diff']:.2e}")
    for k in ("hand_written", "torch_nn"):
        print(f"{k:13s} forward {t[k]['forward_ms']:.3f} ms   train step {t[k]['train_step_ms']:.3f} ms")


if __name__ == "__main__":
    main()
