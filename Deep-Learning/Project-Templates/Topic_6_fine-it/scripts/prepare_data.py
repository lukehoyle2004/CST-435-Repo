"""Build the FineTuneFleet corpus: Banking77 -> eight routing queues, split three ways.

    python -m scripts.prepare_data            # writes data/corpus.csv + docs/data/dataset_stats.json
    python -m scripts.prepare_data --push     # ...and loads it into ftf_documents (needs .env)

Steps, all deterministic (pinned dataset revision, fixed seed):

1. Download Banking77 at a pinned commit (10,003 train / 3,080 test messages).
2. Clean whitespace and map each of the 77 intents to its queue (shared/queues.py).
3. Remove duplicates. Inside the original train split, keep the first copy of a
   message. Then drop any train message whose de-duplication key also appears in
   the official test split -- a duplicate across that line would let the model be
   graded on a message it memorised.
4. Carve a validation split out of train: 10%, stratified by *intent* (not queue)
   so every one of the 77 intents is represented in all three splits.
5. Report class balance and length distributions, in characters and in
   DistilBERT tokens, and verify that no dedup key is shared between splits.

The official test split is never touched by model selection: the sweep picks the
best configuration on validation, and test is scored once per configuration.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from shared.checkpoints import (BASE_MODEL, BASE_MODEL_REVISION, DATASET_REPO,
                                DATASET_REVISION, MAX_LENGTH, SPLIT_SEED)
from shared.queues import INTENT_TO_QUEUE, QUEUE_NAMES
from shared.text import clean, dedup_key, sha256

ROOT = Path(__file__).resolve().parents[1]
CORPUS_CSV = ROOT / "data" / "corpus.csv"
STATS_JSON = ROOT / "docs" / "data" / "dataset_stats.json"
VALIDATION_FRACTION = 0.10


def build() -> tuple[pd.DataFrame, dict]:
    raw = load_dataset(DATASET_REPO, revision=DATASET_REVISION)
    intents = raw["train"].features["label"].names
    assert set(intents) == set(INTENT_TO_QUEUE), "queue mapping must cover every intent exactly"

    frames = []
    for split in ("train", "test"):
        df = raw[split].to_pandas()
        df["intent"] = [intents[i] for i in df["label"]]
        df["text"] = df["text"].map(clean)
        df["source_split"] = split
        frames.append(df[["text", "intent", "source_split"]])
    train, test = frames
    counts = {"raw_train": len(train), "raw_test": len(test)}

    train["key"] = train["text"].map(dedup_key)
    test["key"] = test["text"].map(dedup_key)

    before = len(train)
    train = train.drop_duplicates("key", keep="first")
    counts["dropped_duplicate_within_train"] = before - len(train)

    before = len(test)
    test = test.drop_duplicates("key", keep="first")
    counts["dropped_duplicate_within_test"] = before - len(test)

    before = len(train)
    train = train[~train["key"].isin(set(test["key"]))]
    counts["dropped_train_overlapping_test"] = before - len(train)

    tr, va = train_test_split(train, test_size=VALIDATION_FRACTION, random_state=SPLIT_SEED,
                              stratify=train["intent"])
    tr = tr.assign(split="train")
    va = va.assign(split="validation")
    te = test.assign(split="test")
    df = pd.concat([tr, va, te], ignore_index=True)

    # Leakage guard: a dedup key may belong to one split only.
    owners = df.groupby("key")["split"].nunique()
    assert int((owners > 1).sum()) == 0, "a message appears in more than one split"

    df["queue"] = df["intent"].map(INTENT_TO_QUEUE)
    df["text_sha256"] = df["text"].map(sha256)
    df["n_chars"] = df["text"].str.len()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
    df["n_tokens"] = [len(ids) for ids in tok(df["text"].tolist(), truncation=False)["input_ids"]]

    order = {"train": 0, "validation": 1, "test": 2}
    df = df.sort_values(["split", "queue", "intent", "text"], key=lambda s: s.map(order) if s.name == "split" else s)
    df = df.reset_index(drop=True)
    df.insert(0, "doc_id", np.arange(1, len(df) + 1))
    df = df[["doc_id", "split", "queue", "intent", "text", "text_sha256", "n_chars", "n_tokens"]]

    stats = describe(df, counts)
    return df, stats


def _dist(values: pd.Series) -> dict:
    q = np.percentile(values, [5, 25, 50, 75, 95, 99])
    return {"min": int(values.min()), "p5": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]), "p99": float(q[5]), "max": int(values.max()),
            "mean": round(float(values.mean()), 2)}


def describe(df: pd.DataFrame, counts: dict) -> dict:
    splits = ["train", "validation", "test"]
    balance = {}
    for s in splits:
        part = df[df["split"] == s]
        c = Counter(part["queue"])
        balance[s] = {q: {"n": c[q], "share": round(c[q] / len(part), 4)} for q in QUEUE_NAMES}
    return {
        "dataset": f"{DATASET_REPO}@{DATASET_REVISION}",
        "tokenizer": f"{BASE_MODEL}@{BASE_MODEL_REVISION}",
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "cleaning": counts,
        "rows": {s: int((df["split"] == s).sum()) for s in splits},
        "intents_per_split": {s: int(df.loc[df["split"] == s, "intent"].nunique()) for s in splits},
        "queue_balance": balance,
        "intents_per_queue": {q: int(df.loc[df["queue"] == q, "intent"].nunique()) for q in QUEUE_NAMES},
        "length_chars": {s: _dist(df.loc[df["split"] == s, "n_chars"]) for s in splits},
        "length_tokens": {s: _dist(df.loc[df["split"] == s, "n_tokens"]) for s in splits},
        "max_length": MAX_LENGTH,
        "truncated_messages": int((df["n_tokens"] > MAX_LENGTH).sum()),
        "messages_in_more_than_one_split": 0,
    }


def _round(o):
    if isinstance(o, float):
        return round(o, 2)
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    return o


def push(df: pd.DataFrame, batch: int = 500) -> None:
    from api import db
    rows = df.drop(columns=["doc_id"]).to_dict("records")
    for i in range(0, len(rows), batch):
        db.upsert_documents(rows[i:i + batch])
    print(f"upserted {len(rows)} rows into ftf_documents")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--push", action="store_true", help="also load the corpus into Supabase")
    args = ap.parse_args()

    df, stats = build()
    CORPUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CORPUS_CSV, index=False, lineterminator="\n")
    STATS_JSON.write_text(json.dumps(_round(stats), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("cleaning", "rows", "truncated_messages")}, indent=2))

    if args.push:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        push(df)


if __name__ == "__main__":
    main()
