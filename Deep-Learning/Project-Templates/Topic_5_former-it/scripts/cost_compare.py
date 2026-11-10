"""Turn measured latency into a cost comparison against frontier LLM APIs.

    python -m scripts.cost_compare docs/benchmarks/latency_render.json

Inputs: a latency file written by scripts/benchmark_latency.py (the
``benchmarks_view`` rows from Supabase are preferred; the script falls back to
its own per-call summary) and docs/cost_assumptions.json (prices + assumptions).

For each pretrained archetype it reports
* what 1,000 calls would cost on each frontier model at list price, given the
  same input/output token counts (plus an instruction prompt and some billed
  reasoning tokens, both stated assumptions);
* what 1,000 calls cost on each Render plan *at full utilisation* (one request
  at a time, at the measured mean server time), which is the best case for
  self-hosting -- an idle always-on instance still costs its monthly price;
* the monthly call volume at which a fixed-price instance breaks even with
  each frontier API.

Writes docs/benchmarks/cost_comparison.md (and .json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = {"bert": "encoder (fill-mask)", "gpt2": "decoder (continue)", "t5": "encoder-decoder (summarize)"}


def load_stats(path: Path) -> tuple[dict, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    view = data.get("benchmarks_view")
    if isinstance(view, dict) and view.get("rows"):
        stats = {r["model"]: {"input_tokens": r["mean_input_tokens"], "output_tokens": r["mean_output_tokens"],
                              "server_ms": r["mean_total_ms"], "latency_ms": r["mean_latency_ms"],
                              "n": r["n_calls"]} for r in view["rows"]}
        return stats, "Supabase trio_benchmarks view"
    summ = data["summary"]["grouped"]
    stats = {m: {"input_tokens": None, "output_tokens": s["mean_output_tokens"],
                 "server_ms": s["median_total_ms"], "latency_ms": s["median_latency_ms"], "n": s["n"]}
             for m, s in summ.items()}
    # input token counts per call
    for m in stats:
        calls = [c for c in data["calls"]["grouped"] if c["model"] == m]
        stats[m]["input_tokens"] = sum(c["input_tokens"] for c in calls) / len(calls)
    return stats, "benchmark script summary (database not reachable)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("latency_file", type=Path)
    p.add_argument("--assumptions", type=Path, default=ROOT / "docs" / "cost_assumptions.json")
    args = p.parse_args()
    a = json.loads(args.assumptions.read_text(encoding="utf-8"))
    stats, source = load_stats(args.latency_file)
    fl, sh = a["frontier_llm"], a["self_hosted"]
    seconds_per_month = sh["hours_per_month"] * 3600

    rows = []
    for model in ("bert", "gpt2", "t5"):
        if model not in stats:
            continue
        s = stats[model]
        tin = s["input_tokens"] + fl["prompt_overhead_tokens"]
        tout = s["output_tokens"] + fl["reasoning_tokens_per_call"]
        frontier = {name: 1000 * (tin * pr["input_per_mtok"] + tout * pr["output_per_mtok"]) / 1e6
                    for name, pr in fl["models"].items()}
        capacity = seconds_per_month * 1000 / s["server_ms"]          # calls/month, one at a time
        hosted = {plan: (1000 * cfg["usd_per_month"] / capacity) for plan, cfg in sh["plans"].items()}
        breakeven = {f"{plan} vs {name}": (cfg["usd_per_month"] / (cost / 1000) if cost else None)
                     for plan, cfg in sh["plans"].items() if cfg["usd_per_month"]
                     for name, cost in frontier.items()}
        rows.append({"model": model, "archetype": ARCH[model], **s, "capacity_calls_per_month": capacity,
                     "frontier_usd_per_1k": frontier, "self_hosted_usd_per_1k_at_full_use": hosted,
                     "breakeven_calls_per_month": breakeven})

    names = list(fl["models"])
    plans = [pl for pl, c in sh["plans"].items() if c["usd_per_month"]]
    md = [f"# Cost comparison (generated {a['as_of']})", "",
          f"Latency source: **{source}** (`{args.latency_file.as_posix()}`). "
          f"Assumptions and prices: `docs/cost_assumptions.json` -- {fl['_assumptions']}", "",
          "## Cost per 1,000 calls (USD)", "",
          "| archetype | mean server ms | tokens in / out | " + " | ".join(names) + " | "
          + " | ".join(f"Render {pl} (full use)" for pl in plans) + " |",
          "|" + "---|" * (3 + len(names) + len(plans))]
    for r in rows:
        md.append(f"| {r['archetype']} | {r['server_ms']:.0f} | {r['input_tokens']:.0f} / {r['output_tokens']:.0f} | "
                  + " | ".join(f"${r['frontier_usd_per_1k'][n]:.3f}" for n in names) + " | "
                  + " | ".join(f"${r['self_hosted_usd_per_1k_at_full_use'][pl]:.3f}" for pl in plans) + " |")
    md += ["", "## Break-even volume (calls per month)", "",
           "Below this volume the pay-per-call API is cheaper than keeping the Render instance running.", "",
           "| archetype | " + " | ".join(f"{pl} vs {n}" for pl in plans for n in names) + " | capacity at measured latency |",
           "|" + "---|" * (2 + len(plans) * len(names))]
    for r in rows:
        md.append(f"| {r['archetype']} | " + " | ".join(
            f"{r['breakeven_calls_per_month'][f'{pl} vs {n}']:,.0f}" for pl in plans for n in names)
            + f" | {r['capacity_calls_per_month']:,.0f} |")
    md += ["", "A break-even volume larger than the capacity column means the instance can never serve enough "
           "calls one at a time to beat that API on price.", ""]
    dest = ROOT / "docs" / "benchmarks" / "cost_comparison.md"
    dest.write_text("\n".join(md), encoding="utf-8")
    dest.with_suffix(".json").write_text(json.dumps({"source": source, "rows": rows}, indent=2), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
