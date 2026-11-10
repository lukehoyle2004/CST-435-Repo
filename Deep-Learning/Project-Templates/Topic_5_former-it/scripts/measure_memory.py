"""Measure resident memory (RSS) for each checkpoint, one at a time, as Render will run them.

    python -m scripts.measure_memory [--dtype float32|bfloat16]

Prints RSS after importing torch + transformers and the peak after loading each
model and running one inference, then frees it. Writes docs/benchmarks/memory_<dtype>.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = p.parse_args()
    os.environ["TRIO_DTYPE"] = args.dtype
    base = rss_mb()
    import torch  # noqa: F401
    import transformers  # noqa: F401

    from src import pipelines as P

    torch.set_num_threads(1)
    out = {"dtype": args.dtype, "python_only_mb": round(base, 1), "after_imports_mb": round(rss_mb(), 1), "models": {}}
    text = ("The city council voted on Tuesday to expand the bike lane network after a year of "
            "public meetings. Supporters said the lanes would make streets safer, while some shop "
            "owners worried about losing parking spaces in front of their stores.")
    for name, spec in P.SPECS.items():
        gc.collect()
        before = rss_mb()
        t0 = time.perf_counter()
        bundle = P.load_bundle(spec)
        load_s = time.perf_counter() - t0
        after_load = rss_mb()
        if name == "bert":
            res = P.bert_fill_mask(bundle, P.prepare_masked_text(text, "safer"))
        elif name == "gpt2":
            res = P.gpt2_continue(bundle, text)
        else:
            res = P.t5_summarize(bundle, text)
        peak = rss_mb()
        out["models"][name] = {
            "checkpoint": bundle.checkpoint_id, "params": bundle.params,
            "rss_before_mb": round(before, 1), "rss_after_load_mb": round(after_load, 1),
            "rss_after_inference_mb": round(peak, 1), "load_s": round(load_s, 2),
            "latency_ms": round(res[3], 1), "output": res[0],
        }
        print(name, json.dumps(out["models"][name], ensure_ascii=False))
        del bundle, res
        P.release_memory()
    out["after_unload_mb"] = round(rss_mb(), 1)
    dest = ROOT / "docs" / "benchmarks" / f"memory_{args.dtype}.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "models"}))


if __name__ == "__main__":
    main()
