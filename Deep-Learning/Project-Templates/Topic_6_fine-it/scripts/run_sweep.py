"""Run the zero-shot baseline and the fine-tuning sweep, then score everything on test.

    python -m scripts.run_sweep --pick-prompt     # choose the zero-shot prompt on VALIDATION
    python -m scripts.run_sweep                   # baseline + every config in configs/sweep.yaml
    python -m scripts.run_sweep --resume          # skip configs whose checkpoint already exists
    python -m scripts.run_sweep --push            # also write the rows to Supabase (needs .env)

Outputs (all committed except the weights):
    docs/experiments/results.json              one record per experiment (= ftf_experiments rows)
    docs/experiments/sweep_table.md            the hyperparameter sweep table for the README
    docs/experiments/trainer_logs/<run>.json   the Trainer's full log_history per run
    docs/experiments/test_predictions.csv      every model's test prediction per message
    models/checkpoints/<run>/                  weights (git-ignored; the deployed one ships separately)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from shared.checkpoints import BASE_MODEL, BASE_MODEL_REVISION
from shared.queues import PRIORITY_QUEUE, QUEUE_NAMES, QUEUE_TO_ID
from src import finetune
from src.metrics import evaluate, macro_f1, per_class

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus.csv"
EXP_DIR = ROOT / "docs" / "experiments"
CKPT_DIR = ROOT / "models" / "checkpoints"
DEPLOY_DIR = ROOT / "models" / "deployed"
PROMPT_FILE = EXP_DIR / "zero_shot_prompt_selection.json"
BASELINE_RUN = "zero-shot-flan-t5-small"
PRIORITY = QUEUE_TO_ID[PRIORITY_QUEUE]
LATENCY_SAMPLE = 200


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def ms_per_message(fn, texts: list[str]) -> float:
    """Median single-message latency (batch of one, warm) on this machine, in ms."""
    fn(texts[:3])
    times = []
    for t in texts:
        t0 = time.perf_counter()
        fn([t])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def pick_prompt(splits) -> None:
    """Score every prompt raw and calibrated on VALIDATION; keep the best macro-F1."""
    from src.zero_shot import PROMPTS, ZeroShotRouter
    va = splits["validation"]
    y = va["label"].to_numpy()
    results = {}
    for name in PROMPTS:
        z = ZeroShotRouter(name, calibrate=True)
        raw = z.predict_batch(va["text"].tolist(), calibrated=False)
        for cal, P in (("raw", raw), ("calibrated", z.apply_prior(raw))):
            p = P.argmax(-1).numpy()
            results[f"{name}/{cal}"] = {"val_macro_f1": round(macro_f1(y, p, len(QUEUE_NAMES)), 4),
                                        "val_accuracy": round(float((p == y).mean()), 4),
                                        "predicted_queue_counts": np.bincount(p, minlength=8).tolist()}
            print(name, cal, results[f"{name}/{cal}"])
    best = max(results, key=lambda k: results[k]["val_macro_f1"])
    prompt, cal = best.split("/")
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_FILE.write_text(json.dumps({"chosen": prompt, "calibrate": cal == "calibrated",
                                       "on_split": "validation", "candidates": results}, indent=2) + "\n",
                           encoding="utf-8")
    print("chosen:", best)


def record(run_name, kind, checkpoint, seed, hyper, n_train, val_f1, ev, y, p, runtime, ms, artifact=None):
    return {
        "run_name": run_name, "kind": kind, "base_checkpoint": checkpoint, "seed": seed,
        "hyperparams": hyper, "n_train": n_train, "val_macro_f1": round(val_f1, 4),
        "test_accuracy": round(ev["accuracy"], 4),
        "test_accuracy_lo": round(ev["accuracy_ci"][0], 4), "test_accuracy_hi": round(ev["accuracy_ci"][1], 4),
        "test_macro_f1": round(ev["macro_f1"], 4),
        "test_macro_f1_lo": round(ev["macro_f1_ci"][0], 4), "test_macro_f1_hi": round(ev["macro_f1_ci"][1], 4),
        "fraud_recall": round(ev["priority_recall"], 4),
        "fraud_recall_lo": round(ev["priority_recall_ci"][0], 4),
        "fraud_recall_hi": round(ev["priority_recall_ci"][1], 4),
        "delta_macro_f1": round(ev["delta_macro_f1"], 4) if "delta_macro_f1" in ev else None,
        "delta_macro_f1_lo": round(ev["delta_macro_f1_ci"][0], 4) if "delta_macro_f1" in ev else None,
        "delta_macro_f1_hi": round(ev["delta_macro_f1_ci"][1], 4) if "delta_macro_f1" in ev else None,
        "per_queue": per_class(y, p, QUEUE_NAMES),
        "train_runtime_s": round(runtime, 1), "eval_ms_per_message": round(ms, 2),
        "artifact_sha256": artifact, "is_deployed": False, "git_sha": git_sha(),
    }


def run(resume: bool) -> list[dict]:
    from src.zero_shot import ZeroShotRouter
    splits = finetune.load_splits(CORPUS)
    te = splits["test"]
    y = te["label"].to_numpy()
    texts = te["text"].tolist()
    lat_texts = te.sample(LATENCY_SAMPLE, random_state=0)["text"].tolist()
    preds = pd.DataFrame({"doc_id": te["doc_id"], "true_queue": te["queue"]})

    choice = json.loads(PROMPT_FILE.read_text())
    prompt, calibrate = choice["chosen"], choice["calibrate"]
    print(f"[baseline] flan-t5-small, prompt={prompt}, calibrate={calibrate}")
    z = ZeroShotRouter(prompt, calibrate=calibrate)
    zp = z.predict_batch(texts).argmax(-1).numpy()
    zv = z.predict_batch(splits["validation"]["text"].tolist()).argmax(-1).numpy()
    z_ms = ms_per_message(z.scores, lat_texts)
    ev = evaluate(y, zp, len(QUEUE_NAMES), PRIORITY)
    rows = [record(BASELINE_RUN, "zero_shot", z.checkpoint, 0, {"prompt": prompt, "calibrated": calibrate,
                    "scoring": "label log-likelihood"},
                   0, macro_f1(splits["validation"]["label"].to_numpy(), zv, len(QUEUE_NAMES)),
                   ev, y, zp, 0.0, z_ms)]
    preds[BASELINE_RUN] = [QUEUE_NAMES[i] for i in zp]
    del z

    cfgs = yaml.safe_load((ROOT / "configs" / "sweep.yaml").read_text())["runs"]
    for raw in cfgs:
        cfg = finetune.Config(**raw)
        out = CKPT_DIR / cfg.run_name
        meta_file = out / "ftf_run.json"
        if resume and meta_file.exists():
            print(f"[{cfg.run_name}] resume: loading saved checkpoint")
            meta = json.loads(meta_file.read_text())
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model = AutoModelForSequenceClassification.from_pretrained(out).eval()
            tok = AutoTokenizer.from_pretrained(out)
        else:
            print(f"[{cfg.run_name}] training {cfg.hyperparams()}")
            res = finetune.train_config(cfg, splits, out, EXP_DIR / "trainer_logs")
            model, tok = res["model"], res["tokenizer"]
            meta = {k: res[k] for k in ("val_macro_f1", "train_runtime_s", "n_train")}
            shutil.rmtree(out / "_trainer", ignore_errors=True)
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        proba = finetune.predict_proba(model, tok, texts)
        p = proba.argmax(-1)
        ms = ms_per_message(lambda b: finetune.predict_proba(model, tok, b), lat_texts)
        ev = evaluate(y, p, len(QUEUE_NAMES), PRIORITY, baseline=zp)
        hyper = {**cfg.hyperparams(), "notes": cfg.notes}
        rows.append(record(cfg.run_name, "fine_tuned", f"{BASE_MODEL}@{BASE_MODEL_REVISION}", cfg.seed, hyper,
                           meta["n_train"], meta["val_macro_f1"], ev, y, p, meta["train_runtime_s"], ms,
                           finetune.weights_sha256(out)))
        preds[cfg.run_name] = [QUEUE_NAMES[i] for i in p]
        print(f"[{cfg.run_name}] val F1 {meta['val_macro_f1']:.4f}  test F1 {ev['macro_f1']:.4f} "
              f"delta {ev['delta_macro_f1']:+.4f} {tuple(round(v, 4) for v in ev['delta_macro_f1_ci'])}")

    tuned = [r for r in rows if r["kind"] == "fine_tuned"]
    best = max(tuned, key=lambda r: r["val_macro_f1"])
    best["is_deployed"] = True
    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    shutil.copytree(CKPT_DIR / best["run_name"], DEPLOY_DIR, ignore=shutil.ignore_patterns("_trainer"))
    (DEPLOY_DIR / "ftf_run.json").write_text(json.dumps({"run_name": best["run_name"],
                                                         "artifact_sha256": best["artifact_sha256"]}, indent=2))
    print(f"deployed: {best['run_name']} (best validation macro-F1)")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    preds.to_csv(EXP_DIR / "test_predictions.csv", index=False, lineterminator="\n")
    write_table(rows)
    return rows


def write_table(rows: list[dict]) -> None:
    head = ("| Rank | Run | LR | Epochs | Train msgs | Encoder | Val macro-F1 | Test macro-F1 [95% CI] "
            "| Δ vs zero-shot [95% CI] | Test accuracy | Fraud recall [95% CI] | Train time | ms/msg |\n"
            "|---:|---|---:|---:|---:|---|---:|---|---|---:|---|---:|---:|\n")
    lines = []
    for i, r in enumerate(sorted(rows, key=lambda r: -r["test_macro_f1"]), 1):
        h = r["hyperparams"]
        zs = r["kind"] == "zero_shot"
        delta = "--" if zs else f"{r['delta_macro_f1']:+.3f} [{r['delta_macro_f1_lo']:+.3f}, {r['delta_macro_f1_hi']:+.3f}]"
        name = f"**{r['run_name']}** (deployed)" if r["is_deployed"] else r["run_name"]
        train_time = "--" if zs else f"{r['train_runtime_s'] / 60:.1f} min"
        lines.append(
            f"| {i} | {name} | {'--' if zs else h['learning_rate']} | {'--' if zs else h['epochs']} "
            f"| {r['n_train']:,} | {'zero-shot' if zs else ('frozen' if h['freeze_encoder'] else 'trained')} "
            f"| {r['val_macro_f1']:.3f} | {r['test_macro_f1']:.3f} [{r['test_macro_f1_lo']:.3f}, {r['test_macro_f1_hi']:.3f}] "
            f"| {delta} | {r['test_accuracy']:.3f} | {r['fraud_recall']:.3f} [{r['fraud_recall_lo']:.3f}, {r['fraud_recall_hi']:.3f}] "
            f"| {train_time} | {r['eval_ms_per_message']:.1f} |")
    (EXP_DIR / "sweep_table.md").write_text(
        "<!-- generated by scripts/run_sweep.py from docs/experiments/results.json -->\n"
        + head + "\n".join(lines) + "\n", encoding="utf-8")


def push(rows: list[dict]) -> None:
    from dotenv import load_dotenv
    from api import db
    load_dotenv(ROOT / ".env")
    db.get_client().table("ftf_experiments").update({"is_deployed": False}).eq("is_deployed", True).execute()
    for r in rows:
        db.upsert_experiment(r)
    db.set_deployed(next(r["run_name"] for r in rows if r["is_deployed"]))
    print(f"upserted {len(rows)} experiments")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pick-prompt", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--push", action="store_true", help="write results.json to Supabase (no training)")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    if args.pick_prompt:
        pick_prompt(finetune.load_splits(CORPUS))
    elif args.push:
        push(json.loads((EXP_DIR / "results.json").read_text()))
    else:
        run(args.resume)


if __name__ == "__main__":
    main()
