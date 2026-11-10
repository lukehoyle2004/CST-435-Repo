"""Drive the deployed (or local) API and collect latency evidence for the report.

    python -m scripts.benchmark_latency --base-url https://<your-api>.onrender.com --rounds 5

For each round it sends three paragraphs (short / medium / long) to every model
endpoint, tagged ``source="benchmark"`` so the rows are identifiable in Supabase.
Two orders are run, because on a one-model-at-a-time instance they tell
different cost stories:

* ``grouped``      all calls for one model back to back -> mostly warm
* ``interleaved``  BERT, GPT-2, T5, BERT, ... -> every pretrained call is a cold swap
  (the "Compare all three" pattern a 5B interface produces)

Writes ``docs/benchmarks/latency_<label>.json`` with every call (server-side
latency_ms / load_ms / total_ms and the client round trip) plus the
``/benchmarks`` materialized-view rows fetched at the end, when available.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

PARAGRAPHS = {
    "short": ("The library will close early on Friday for staff training. Students who need to "
              "return books can use the drop box by the main entrance, which is emptied every morning."),
    "medium": ("The city council voted on Tuesday to expand the bike lane network after a year of public "
               "meetings. Supporters said the new lanes would make downtown streets safer for commuters, "
               "while several shop owners worried about losing parking in front of their stores. The plan "
               "adds twelve miles of protected lanes over three years and will be paid for with a mix of "
               "state grants and the city's road maintenance budget."),
    "long": ("A small software team spent the spring replacing an aging scheduling system that had been "
             "patched for more than a decade. The old system stored every appointment in a single table, "
             "and over time the table grew so large that simple searches took several seconds. The team "
             "began by interviewing the front-desk staff who used the tool every day, and learned that "
             "most of their frustration came from three tasks: rescheduling, finding open slots, and "
             "printing the daily list. Instead of rebuilding everything at once, the developers rewrote "
             "those three features first and ran them beside the old system for a month. Staff could "
             "switch back whenever something felt wrong, and their feedback shaped weekly releases. By "
             "summer the new system handled all scheduling, searches returned almost instantly, and the "
             "team had a written record of every decision they made along the way."),
}
MASK_WORD = {"short": "early", "medium": "safer", "long": "frustration"}
ENCODER_INPUTS = ["dog bites man bites dog", "the cat sat on the mat", "never odd or even odd never"]


def call(session: requests.Session, base: str, model: str, size: str) -> dict:
    text = PARAGRAPHS[size]
    if model == "encoder_block":
        path, payload = "/encoder_attention", {"text": ENCODER_INPUTS[len(size) % 3]}
    elif model == "bert":
        path, payload = "/bert_fillmask", {"text": text, "mask_word": MASK_WORD[size]}
    elif model == "gpt2":
        path, payload = "/gpt2_continue", {"text": text, "max_new_tokens": 40}
    else:
        path, payload = "/t5_summarize", {"text": text, "max_new_tokens": 60}
    payload["source"] = "benchmark"
    t0 = time.perf_counter()
    r = session.post(base + path, json=payload, timeout=300)
    rtt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    body = r.json()
    return {k: body[k] for k in ("model", "latency_ms", "load_ms", "total_ms", "cold_start",
                                 "input_tokens", "output_tokens", "logged")} | {"size": size, "client_rtt_ms": round(rtt, 1)}


def summarise(calls: list[dict]) -> dict:
    out = {}
    for model in ("encoder_block", "bert", "gpt2", "t5"):
        rows = [c for c in calls if c["model"] == model]
        if not rows:
            continue
        warm = [c["latency_ms"] for c in rows if not c["cold_start"]]
        cold = [c["load_ms"] for c in rows if c["cold_start"]]
        out[model] = {
            "n": len(rows), "n_cold": len(cold),
            "median_latency_ms": round(statistics.median(c["latency_ms"] for c in rows), 1),
            "median_warm_latency_ms": round(statistics.median(warm), 1) if warm else None,
            "median_load_ms_when_cold": round(statistics.median(cold), 1) if cold else None,
            "median_total_ms": round(statistics.median(c["total_ms"] for c in rows), 1),
            "median_client_rtt_ms": round(statistics.median(c["client_rtt_ms"] for c in rows), 1),
            "mean_output_tokens": round(statistics.mean(c["output_tokens"] for c in rows), 1),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--label", default=None, help="file suffix; default derived from the URL")
    args = p.parse_args()
    base = args.base_url.rstrip("/")
    label = args.label or ("render" if "onrender.com" in base else "local")
    session = requests.Session()

    print("waking the service ...")
    health = session.get(base + "/healthz", timeout=180).json()
    version = session.get(base + "/version", timeout=60).json()

    results = {"grouped": [], "interleaved": []}
    for r in range(args.rounds):
        for model in ("encoder_block", "bert", "gpt2", "t5"):
            for size in PARAGRAPHS:
                results["grouped"].append(call(session, base, model, size))
        for size in PARAGRAPHS:
            for model in ("bert", "gpt2", "t5"):
                results["interleaved"].append(call(session, base, model, size))
        print(f"round {r + 1}/{args.rounds} done")

    bench = session.get(base + "/benchmarks", timeout=60)
    out = {
        "base_url": base, "rounds": args.rounds, "started_health": health, "version": version,
        "summary": {order: summarise(calls) for order, calls in results.items()},
        "calls": results,
        "benchmarks_view": bench.json() if bench.ok else {"error": bench.status_code},
    }
    dest = ROOT / "docs" / "benchmarks" / f"latency_{label}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
