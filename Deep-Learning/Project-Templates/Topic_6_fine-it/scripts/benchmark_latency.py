"""Measure the deployed service and price it against frontier LLM APIs.

    python -m scripts.benchmark_latency --base-url https://finetunefleet-api.onrender.com

What it does, all against the live service (rows are logged with source='benchmark'
or 'api', so they also feed the ftf_latency view):

1. Warm single calls: 10 x /predict in a row, then 10 x /predict_zero_shot in a row.
   The first call of each block pays the model load (the 512 MB instance keeps one
   model resident), so it is reported separately as the switch cost.
2. Batches: the fixed 16-message batch through /predict_batch for each model,
   ``--rounds`` times.

Writes docs/benchmarks/latency_render.json and docs/benchmarks/cost_comparison.md,
pricing the measured fine-tuned throughput against docs/cost_assumptions.json.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks"
ASSUME = json.loads((ROOT / "docs" / "cost_assumptions.json").read_text())
BATCH = json.loads((ROOT / "tests" / "fixtures" / "regression.json").read_text(encoding="utf-8"))["cases"]
BATCH = [c["text"] for c in BATCH][:16]
SINGLE = "Someone used my card without my permission."


def call(base: str, path: str, payload: dict | None = None) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = requests.post(base + path, json=payload, timeout=300) if payload is not None else \
        requests.get(base + path, timeout=300)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - t0) * 1000


def summarize(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"n": len(xs), "p50": round(statistics.median(xs), 1),
            "p95": round(xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)], 1),
            "mean": round(statistics.fmean(xs), 1)}


def measure(base: str, rounds: int) -> dict:
    health, _ = call(base, "/healthz")
    version, _ = call(base, "/version")
    out = {"base_url": base, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "git_sha": version.get("git_sha"), "deployed_run": health.get("deployed_run"), "single": {}, "batch": {}}
    for kind, path in (("fine_tuned", "/predict"), ("zero_shot", "/predict_zero_shot")):
        rows = [call(base, path, {"text": SINGLE}) for _ in range(10)]
        first, rest = rows[0], rows[1:]
        out["single"][kind] = {
            "switch_load_ms": first[0]["load_ms"], "switch_e2e_ms": round(first[1], 1),
            "warm_server_ms": summarize([d["latency_ms"] for d, _ in rest]),
            "warm_e2e_ms": summarize([e for _, e in rest]),
            "n_tokens": first[0]["n_tokens"], "logged": sum(d["logged"] for d, _ in rows),
        }
        print(kind, out["single"][kind])
    for kind in ("fine_tuned", "zero_shot"):
        runs = []
        for _ in range(rounds):
            d, e2e = call(base, "/predict_batch", {"texts": BATCH, "model": kind})
            runs.append({"ms_per_message": d["total_latency_ms"] / len(BATCH), "wall_ms": d["wall_ms"],
                         "e2e_ms": e2e, "load_ms": sum(p["load_ms"] for p in d["predictions"]),
                         "mean_tokens": statistics.fmean(p["n_tokens"] for p in d["predictions"])})
        out["batch"][kind] = {"rounds": runs, "ms_per_message": summarize([r["ms_per_message"] for r in runs]),
                              "mean_tokens": round(runs[0]["mean_tokens"], 1)}
        print(kind, out["batch"][kind]["ms_per_message"])
    return out


def price(m: dict) -> str:
    fr, sh = ASSUME["frontier_llm"], ASSUME["self_hosted"]
    ms_ft = m["batch"]["fine_tuned"]["ms_per_message"]["p50"]
    ms_zs = m["batch"]["zero_shot"]["ms_per_message"]["p50"]
    tokens = m["batch"]["fine_tuned"]["mean_tokens"]
    in_tok = fr["prompt_tokens"] + tokens
    secs = sh["hours_per_month"] * 3600
    base_cpu = sh["plans"][sh["measured_on"]]["cpu"]
    lines = [f"# Cost comparison: self-hosted router vs frontier LLM APIs",
             "",
             f"Measured {m['measured_at']} on `{m['base_url']}` (commit `{(m['git_sha'] or '')[:7]}`), Render free "
             f"instance (0.1 CPU, 512 MB). Prices as of {ASSUME['as_of']}; every input is in "
             f"[`docs/cost_assumptions.json`](../cost_assumptions.json).",
             "",
             "## Measured on the live server",
             "",
             "| | Fine-tuned DistilBERT | Zero-shot flan-t5-small |",
             "|---|---:|---:|",
             f"| Server model time per message, batch of 16 (p50 of {len(m['batch']['fine_tuned']['rounds'])} rounds) | "
             f"{ms_ft:,.0f} ms | {ms_zs:,.0f} ms |",
             f"| Single call, warm: server model time p50 | {m['single']['fine_tuned']['warm_server_ms']['p50']:,.0f} ms | "
             f"{m['single']['zero_shot']['warm_server_ms']['p50']:,.0f} ms |",
             f"| Single call, warm: end-to-end p50 (laptop ↔ Render) | {m['single']['fine_tuned']['warm_e2e_ms']['p50']:,.0f} ms | "
             f"{m['single']['zero_shot']['warm_e2e_ms']['p50']:,.0f} ms |",
             f"| Model load when switching to it (one model resident) | {m['single']['fine_tuned']['switch_load_ms']:,.0f} ms | "
             f"{m['single']['zero_shot']['switch_load_ms']:,.0f} ms |",
             "| Test macro-F1 [95% CI] | **0.965** [0.958, 0.972] | 0.669 [0.649, 0.686] |",
             "",
             "## Monthly cost by volume",
             "",
             f"Frontier calls: an instruction listing the eight queues (~{fr['prompt_tokens']} tokens) plus the message "
             f"(measured mean {tokens:.1f} DistilBERT tokens, a ±30% proxy for the API's tokenizer), "
             f"{fr['output_tokens']} output tokens, plus each model's billed reasoning tokens. Self-hosted capacity = "
             "seconds per month ÷ measured seconds per message, scaled by the plan's CPU share (assumes CPU-bound).",
             "",
             "| Option | per 1,000 msgs | 10k / month | 100k / month | 1M / month | 10M / month |",
             "|---|---:|---:|---:|---:|---:|"]
    vols = [10_000, 100_000, 1_000_000, 10_000_000]
    for plan, p in sh["plans"].items():
        ms = ms_ft * base_cpu / p["cpu"]
        cap = secs * 1000 / ms
        cells = []
        for v in vols:
            n = max(1, math.ceil(v / cap))
            cells.append(f"${n * p['usd_per_month']:,.0f}" + (f" ({n}×)" if n > 1 else ""))
        per_k = p["usd_per_month"] / min(cap, vols[-1]) * 1000 if p["usd_per_month"] else 0
        lines.append(f"| Self-hosted, Render {plan} ({p['cpu']} CPU; capacity ≈ {cap / 1e6:,.1f}M msgs/month) | "
                     f"${per_k:,.4f} at capacity | " + " | ".join(cells) + " |")
    for name, p in fr["models"].items():
        per = (in_tok * p["input_per_mtok"] + (fr["output_tokens"] + p["reasoning_tokens"]) * p["output_per_mtok"]) / 1e6
        lines.append(f"| {name} API | ${per * 1000:,.3f} | " + " | ".join(f"${per * v:,.0f}" for v in vols) + " |")
    lines += ["",
              "Frontier quality was **not** measured (sending the test set to a paid API was out of scope); the "
              "comparison is cost and speed only for those rows. Engineering time to build and operate the "
              "self-hosted pipeline is not included in its column, and it is the real cost at low volume.",
              ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://finetunefleet-api.onrender.com")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reprice", action="store_true", help="re-price the saved measurement only")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "latency_render.json"
    m = json.loads(f.read_text()) if a.reprice else measure(a.base_url.rstrip("/"), a.rounds)
    f.write_text(json.dumps(m, indent=2) + "\n")
    (OUT / "cost_comparison.md").write_text(price(m), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print((OUT / "cost_comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
