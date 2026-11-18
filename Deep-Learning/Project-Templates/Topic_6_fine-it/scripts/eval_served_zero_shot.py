"""Check that the zero-shot model the service runs is the one the leaderboard scored.

    python -m scripts.eval_served_zero_shot

To fit Render's 512 MB instance, the service keeps flan-t5-small's weights in
float16 (computing in float32; src/zero_shot.py). The leaderboard row
``zero-shot-flan-t5-small`` was scored in float32. This script scores the whole
test split with the served variant and records how often the two agree and the
served variant's own macro-F1 with a 95% bootstrap CI, in
docs/experiments/zero_shot_served_variant.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from shared.queues import PRIORITY_QUEUE, QUEUE_NAMES, QUEUE_TO_ID
from src.metrics import evaluate
from src.zero_shot import ZeroShotRouter

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs" / "experiments"


def main() -> None:
    choice = json.loads((EXP / "zero_shot_prompt_selection.json").read_text())
    corpus = pd.read_csv(ROOT / "data" / "corpus.csv")
    test = corpus[corpus["split"] == "test"].reset_index(drop=True)
    saved = pd.read_csv(EXP / "test_predictions.csv").set_index("doc_id")["zero-shot-flan-t5-small"]
    fp32_pred = saved.loc[test["doc_id"]].tolist()

    z = ZeroShotRouter(choice["chosen"], calibrate=choice["calibrate"], half_storage=True)
    t0 = time.perf_counter()
    probs = z.predict_batch(test["text"].tolist(), batch_size=4)
    ms = (time.perf_counter() - t0) * 1000 / len(test)
    pred = [QUEUE_NAMES[i] for i in probs.argmax(-1).tolist()]

    y = test["queue"].map(QUEUE_TO_ID).to_numpy()
    p = np.array([QUEUE_TO_ID[q] for q in pred])
    p32 = np.array([QUEUE_TO_ID[q] for q in fp32_pred])
    served = evaluate(y, p, len(QUEUE_NAMES), QUEUE_TO_ID[PRIORITY_QUEUE])
    scored = evaluate(y, p32, len(QUEUE_NAMES), QUEUE_TO_ID[PRIORITY_QUEUE])
    out = {
        "served_checkpoint": z.checkpoint,
        "leaderboard_run": "zero-shot-flan-t5-small",
        "n_test": len(test),
        "agreement_with_float32": round(float((p == p32).mean()), 4),
        "n_disagreements": int((p != p32).sum()),
        "served_test_macro_f1": round(served["macro_f1"], 4),
        "served_test_macro_f1_ci": [round(v, 4) for v in served["macro_f1_ci"]],
        "float32_test_macro_f1": round(scored["macro_f1"], 4),
        "float32_test_macro_f1_ci": [round(v, 4) for v in scored["macro_f1_ci"]],
        "served_ms_per_message_local": round(ms, 1),
        "why": "float16-stored weights (float32 compute) halve flan-t5-small's memory so it fits "
               "Render's 512 MB instance; only the stored weights are rounded.",
    }
    (EXP / "zero_shot_served_variant.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
